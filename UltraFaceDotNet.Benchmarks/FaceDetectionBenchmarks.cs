using BenchmarkDotNet.Attributes;
using SkiaSharp;

namespace UltraFaceDotNet.Benchmarks;

[MemoryDiagnoser]
public class FaceDetectionBenchmarks
{
    private FaceDetector _faceDetector;
    private string _modelPath;
    private SKBitmap _bitmap;
    public FaceDetectionBenchmarks()
    {
        _modelPath = @"D:\repos\net7\UltraFaceDotNet\UltraFaceDotNet.Benchmarks\Assets\version-RFB-640.onnx";
        _faceDetector = new FaceDetector(_modelPath, 0.5F, 0.75F);
        
        _bitmap = SKBitmap.Decode(@"D:\repos\net7\UltraFaceDotNet\UltraFaceDotNet.Benchmarks\Assets\face.jpg");
        
        _faceDetector.Initialize();
    }
    
    [Benchmark()]
    public void DetectFaces() => _faceDetector.Detect(_bitmap);
}