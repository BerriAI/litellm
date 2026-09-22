import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

guard CommandLine.arguments.count == 3 else {
    fputs("usage: fill-logo-background.swift input.jpg output.png\n", stderr)
    exit(2)
}

let inputURL = URL(fileURLWithPath: CommandLine.arguments[1])
let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
let targetColor = (red: UInt8(135), green: UInt8(206), blue: UInt8(234))

guard let source = CGImageSourceCreateWithURL(inputURL as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
    fputs("could not read logo\n", stderr)
    exit(1)
}

let imageWidth = image.width
let imageHeight = image.height
let canvasSize = max(imageWidth, imageHeight)
let scale = 1.16
let scaledWidth = Int((Double(imageWidth) * scale).rounded())
let scaledHeight = Int((Double(imageHeight) * scale).rounded())
let origin = CGPoint(x: (canvasSize - scaledWidth) / 2, y: (canvasSize - scaledHeight) / 2)
let colorSpace = CGColorSpaceCreateDeviceRGB()
var pixels = [UInt8](repeating: 0, count: canvasSize * canvasSize * 4)

guard let context = CGContext(
    data: &pixels,
    width: canvasSize,
    height: canvasSize,
    bitsPerComponent: 8,
    bytesPerRow: canvasSize * 4,
    space: colorSpace,
    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
) else {
    fputs("could not create image context\n", stderr)
    exit(1)
}

context.setFillColor(red: CGFloat(targetColor.red) / 255, green: CGFloat(targetColor.green) / 255, blue: CGFloat(targetColor.blue) / 255, alpha: 1)
context.fill(CGRect(x: 0, y: 0, width: canvasSize, height: canvasSize))
context.draw(image, in: CGRect(origin: origin, size: CGSize(width: scaledWidth, height: scaledHeight)))

func isBackgroundPixel(_ offset: Int) -> Bool {
    let red = pixels[offset]
    let green = pixels[offset + 1]
    let blue = pixels[offset + 2]
    let brightest = max(red, max(green, blue))
    let darkest = min(red, min(green, blue))
    return brightest >= 225 && brightest - darkest <= 24
}

var background = [Bool](repeating: false, count: canvasSize * canvasSize)
var queue: [Int] = []

func seed(_ index: Int) {
    guard !background[index], isBackgroundPixel(index * 4) else { return }
    background[index] = true
    queue.append(index)
}

for x in 0..<canvasSize {
    seed(x)
    seed((canvasSize - 1) * canvasSize + x)
}
for y in 0..<canvasSize {
    seed(y * canvasSize)
    seed(y * canvasSize + canvasSize - 1)
}

var cursor = 0
while cursor < queue.count {
    let index = queue[cursor]
    cursor += 1
    let x = index % canvasSize
    let y = index / canvasSize
    if x > 0 { seed(index - 1) }
    if x + 1 < canvasSize { seed(index + 1) }
    if y > 0 { seed(index - canvasSize) }
    if y + 1 < canvasSize { seed(index + canvasSize) }
}

for index in queue {
    let offset = index * 4
    pixels[offset] = targetColor.red
    pixels[offset + 1] = targetColor.green
    pixels[offset + 2] = targetColor.blue
    pixels[offset + 3] = 255
}

guard let result = context.makeImage(),
      let destination = CGImageDestinationCreateWithURL(outputURL as CFURL, UTType.png.identifier as CFString, 1, nil) else {
    fputs("could not write logo\n", stderr)
    exit(1)
}
CGImageDestinationAddImage(destination, result, nil)
guard CGImageDestinationFinalize(destination) else {
    fputs("could not finalize logo\n", stderr)
    exit(1)
}
