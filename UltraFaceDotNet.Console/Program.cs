// See https://aka.ms/new-console-template for more information

using SkiaSharp;
using UltraFaceDotNet;

var assetsPath = Path.Combine(AppContext.BaseDirectory, "Assets");
var modelPath = Path.Combine(assetsPath, "version-RFB-640.onnx");
var imagePath = Path.Combine(assetsPath, "face.jpg");

var faceDetector = new FaceDetector(modelPath, 0.5F, 0.75F);

faceDetector.Initialize();

var bitmap = SKBitmap.Decode(imagePath);

var count = 256;
while (count-- > 0)
{
    _ = faceDetector.Detect(bitmap);
}

faceDetector.Dispose();

Console.WriteLine("Done");
Console.ReadLine();