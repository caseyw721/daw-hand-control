// Camera floor probe: how long the camera takes to deliver a frame, and what it is doing with
// exposure. Needs no hand and no Python. Run it whenever the setup changes (room light, another
// camera) to know the part of motion-to-photon that no software below the camera can fix:
//
//     swift camera-bench/tools/cam_probe.swift [seconds, default 6] [part of a camera name, default: system default]
//
// It prints the delivered frame size, the delivery latency (host clock at the delegate callback
// minus the frame's presentation timestamp, both host time), and the exposure time, ISO and frame
// rate read from each frame's metadata attachment. The frame duration is pinned to 1/30 s the way
// the bench pins it, so auto-exposure cannot fall back to 15 fps; the pin is applied after the
// session starts because applied before, the session's own start-up configuration undid it once
// (15 fps, 66 ms exposures, in a dark room).
//
// On the MacBook Pro camera the session preset and the device's activeFormat are both ignored:
// 1920x1080 arrives whatever is asked, so there is no resolution to choose here.
import AVFoundation
import Foundation

let seconds = CommandLine.arguments.count > 1 ? Double(CommandLine.arguments[1]) ?? 6 : 6

func pct(_ a: [Double], _ p: Double) -> Double {
    if a.isEmpty { return .nan }
    let s = a.sorted()
    return s[min(s.count - 1, Int(p * Double(s.count)))]
}

func dbl(_ v: Any?) -> Double? {
    if let d = v as? Double { return d }
    if let s = v as? String { return Double(s) }
    if let n = v as? NSNumber { return n.doubleValue }
    return nil
}

func fcc(_ u: UInt32) -> String {
    String(bytes: [UInt8(u >> 24 & 255), UInt8(u >> 16 & 255), UInt8(u >> 8 & 255), UInt8(u & 255)], encoding: .ascii) ?? "\(u)"
}

final class Delegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    var lat: [Double] = [], gaps: [Double] = [], expo: [Double] = [], iso: [Double] = [], rates: [Double] = []
    var frames = 0, dropped = 0, dims = "", lastPTS = -1.0
    let warmup = 45  // frames to skip while auto-exposure settles

    func captureOutput(_ output: AVCaptureOutput, didOutput sb: CMSampleBuffer, from connection: AVCaptureConnection) {
        frames += 1
        let pts = CMSampleBufferGetPresentationTimeStamp(sb).seconds
        if frames <= warmup { lastPTS = pts; return }
        let now = CMClockGetTime(CMClockGetHostTimeClock()).seconds
        lat.append((now - pts) * 1000)
        gaps.append((pts - lastPTS) * 1000)
        lastPTS = pts
        if let pb = CMSampleBufferGetImageBuffer(sb) {
            dims = "\(CVPixelBufferGetWidth(pb))x\(CVPixelBufferGetHeight(pb)) \(fcc(CVPixelBufferGetPixelFormatType(pb)))"
        }
        let att = CMCopyDictionaryOfAttachments(allocator: nil, target: sb, attachmentMode: kCMAttachmentMode_ShouldPropagate) as? [String: Any] ?? [:]
        if let md = att["MetadataDictionary"] as? [String: Any] {
            if let e = dbl(md["ExposureTime"]) { expo.append(e * 1000) }
            if let i = dbl(md["ISOSpeedRating"]) { iso.append(i) }
            if let r = dbl(md["CurrentFrameRate"]) { rates.append(r) }
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didDrop sb: CMSampleBuffer, from connection: AVCaptureConnection) {
        dropped += 1
    }
}

let cameras = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera],
                                               mediaType: .video, position: .unspecified).devices
let wanted = CommandLine.arguments.count > 2 ? CommandLine.arguments[2].lowercased() : nil
guard let dev = wanted == nil ? AVCaptureDevice.default(for: .video) : cameras.first(where: { $0.localizedName.lowercased().contains(wanted!) }) else {
    fatalError("no camera matching \(wanted ?? "default"); have: \(cameras.map { $0.localizedName })")
}
print("cameras: \(cameras.map { $0.localizedName })")
print("using: \(dev.localizedName); formats: " + dev.formats.map { f in
    let d = f.formatDescription.dimensions
    let r = f.videoSupportedFrameRateRanges.map { String(format: "%.0f-%.0f", $0.minFrameRate, $0.maxFrameRate) }.joined(separator: "/")
    return "\(d.width)x\(d.height)@\(r)"
}.joined(separator: " "))

let session = AVCaptureSession()
session.beginConfiguration()
let input = try! AVCaptureDeviceInput(device: dev)
session.addInput(input)
let out = AVCaptureVideoDataOutput()
out.alwaysDiscardsLateVideoFrames = true
out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]  // what OpenCV asks for
let d = Delegate()
out.setSampleBufferDelegate(d, queue: DispatchQueue(label: "cam"))
session.addOutput(out)
session.commitConfiguration()
session.startRunning()
// Pin only when the active format allows 1/30 s: an external camera that doesn't (the DJI
// Osmo Action 4 webcam) throws an Objective-C exception from the setter, which ends the probe.
let want = CMTime(value: 1, timescale: 30)
if dev.activeFormat.videoSupportedFrameRateRanges.contains(where: {
    CMTimeCompare($0.minFrameDuration, want) <= 0 && CMTimeCompare(want, $0.maxFrameDuration) <= 0
}) {
    try! dev.lockForConfiguration()
    dev.activeVideoMaxFrameDuration = want  // max first: min may not exceed it
    dev.activeVideoMinFrameDuration = want
    dev.unlockForConfiguration()
} else {
    print("frame duration not pinned: the active format allows " + dev.activeFormat.videoSupportedFrameRateRanges.map {
        String(format: "%.3f-%.3f fps", $0.minFrameRate, $0.maxFrameRate) }.joined(separator: ", "))
}
RunLoop.main.run(until: Date(timeIntervalSinceNow: seconds))
session.stopRunning()

print(String(format: "delivered %@, frame duration pinned %.4f-%.4f s", d.dims, dev.activeVideoMinFrameDuration.seconds, dev.activeVideoMaxFrameDuration.seconds))
print(String(format: "delivery latency ms: p10 %.0f  p50 %.0f  p90 %.0f   (frame gap p50 %.1f ms, %d frames after warm-up, %d dropped)",
             pct(d.lat, 0.1), pct(d.lat, 0.5), pct(d.lat, 0.9), pct(d.gaps, 0.5), d.lat.count, d.dropped))
print(String(format: "exposure ms p50 %.1f max %.1f | ISO p50 %.0f | camera reports %.1f fps", pct(d.expo, 0.5), d.expo.max() ?? .nan, pct(d.iso, 0.5), pct(d.rates, 0.5)))
if pct(d.expo, 0.5) >= 30 {
    print("verdict: exposure sits at the whole frame period, so the room is dark for this camera. Light the hand: a shorter exposure means less motion blur (steadier landmarks, fewer drop-outs) and the frame is captured earlier in its period.")
} else if pct(d.expo, 0.5) >= 15 {
    print("verdict: exposure is long (\(Int(pct(d.expo, 0.5))) ms). More light on the hand would sharpen landmarks during fast moves.")
} else {
    print("verdict: exposure is short; light is not the limit.")
}
if pct(d.rates, 0.5) < 29 {
    print("verdict: the camera is NOT at 30 fps. The pin failed; every frame costs \(Int(pct(d.gaps, 0.5))) ms.")
}
