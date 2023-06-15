using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace UltraFaceDotNet;

public class FaceDetector : IDisposable
{
    private readonly string _modelPath; // Ścieżka do modelu
    private int _modelWidth; // Szerokość obrazu wejściowego
    private int _modelHeight; // Wysokość obrazu wejściowego
    private readonly float _confidenceThreshold; // Minimalne prawdopodobieństwo do uwzględnienia wyniku
    private readonly float _iouThreshold; // Minimalne przecięcie do uwzględnienia wyniku

    private InferenceSession _session = null!; // Sesja ONNX
    
    private string _inputName = null!; // Nazwa tensora wejściowego
    private DenseTensor<float> _inputTensor = null!; // Tensor wejściowy
    
    private string _boxesName = null!; // Nazwa tensora wyjściowego z obszarami zawierającymi twarze
    private DenseTensor<float> _boxesTensor = null!; // Tensor wyjściowy z obszarami zawierającymi twarze
    
    private string _scoresName = null!; // Nazwa tensora wyjściowego z prawdopodobieństwami klas
    private DenseTensor<float> _scoresTensor = null!; // Tensor wyjściowy z prawdopodobieństwami klas
    private List<NamedOnnxValue> _onnxInputs = null!;
    private List<NamedOnnxValue> _onnxOutputs = null!;

    public FaceDetector(string modelPath, float confidenceThreshold = 0.75F, float iouThreshold = 0.5F)
    {
        _modelPath = modelPath;
        _confidenceThreshold = confidenceThreshold;
        _iouThreshold = iouThreshold;
    }

    public void Initialize()
    {
        var fullPath = Path.GetFullPath(_modelPath);
        
        _session = new InferenceSession(fullPath);
        
        // Inicjalizacja wejścia
        var inputMeta = _session.InputMetadata.First();
        var inputDimensions = inputMeta.Value.Dimensions; 
        _inputName = inputMeta.Key;
        _modelHeight = inputDimensions[2];
        _modelWidth = inputDimensions[3];
        
        _inputTensor = new DenseTensor<float>(inputDimensions);
        
        // Inicjalizacja wyjścia
        var outputMeta = _session.OutputMetadata;
        // Region z twarzami
        _boxesName = outputMeta.First(x => x.Key == "boxes").Key;
        var boxesMeta = outputMeta.First(x => x.Key == _boxesName);
        var boxesDimensions = boxesMeta.Value.Dimensions;
        
        _boxesTensor = new DenseTensor<float>(boxesDimensions);
        
        // Prawdopodobieństwa klas
        _scoresName = outputMeta.First(x => x.Key == "scores").Key;
        var scoresMeta = outputMeta.First(x => x.Key == _scoresName);
        var scoresDimensions = scoresMeta.Value.Dimensions;
        
        _scoresTensor = new DenseTensor<float>(scoresDimensions);
        
        
        _onnxInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, _inputTensor)
        };
        
        _onnxOutputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_boxesName, _boxesTensor), NamedOnnxValue.CreateFromTensor(_scoresName, _scoresTensor) };
    }

    public IEnumerable<Face> Detect(SKBitmap bitmap)
    {
        // Obraz wejściowy musi być w rozmiarze modelu
        using var resizedImage = bitmap.Resize(new SKImageInfo(_modelWidth, _modelHeight), SKFilterQuality.High);
        
        resizedImage.ExtractPixelsRgbToTensorSpan(_inputTensor);
        
        _session.Run(_onnxInputs, _onnxOutputs);

        var boxesBuffer = _boxesTensor.Buffer;
        
        var scoresBuffer = _scoresTensor.Buffer;
        
        var unfilteredFaces = ParsePredictions(boxesBuffer, scoresBuffer, bitmap.Width, bitmap.Height, _confidenceThreshold);
        var suppressedFaces = NonMaximumSuppression(unfilteredFaces, _iouThreshold);
        
        return suppressedFaces;
    }

    private static List<Face> ParsePredictions(Memory<float> boxesArray, Memory<float> scoresArray,
        int photoWidth, int photoHeight, float confidenceThreshold = 0.75f)
    {
        var detectedCount = scoresArray.Length / 2;
        var boxesSpan = boxesArray.Span;
        var scoresSpan = scoresArray.Span;
        var detectionResults = new List<Face>(detectedCount);
        
        for (var i = 0; i < detectedCount; i++)
        {
            var backgroundScore = scoresSpan[i * 2];
            var score = scoresSpan[i * 2 + 1];
            
            if (score <= confidenceThreshold) continue; // odfiltruj wyniki poniżej/lub równe progu
            if (backgroundScore >= score) continue; // odfiltruj wyniki z tłem
            
            var left = boxesSpan[i * 4]; // procent szerokości zdjęcia
            var top = boxesSpan[i * 4 + 1]; // procent wysokości zdjęcia
            var right = boxesSpan[i * 4 + 2]; // procent szerokości zdjęcia
            var bottom = boxesSpan[i * 4 + 3]; // procent wysokości zdjęcia

            // zeskaluj do rozmiaru zdjęcia
            var leftClamped = Math.Clamp(left * photoWidth, 0, photoWidth);
            var topClamped = Math.Clamp(top * photoHeight, 0, photoHeight);
            var rightClamped = Math.Clamp(right * photoWidth, 0, photoWidth);
            var bottomClamped = Math.Clamp(bottom * photoHeight, 0, photoHeight);

            var box = new SKRect(leftClamped, topClamped, rightClamped, bottomClamped);
            detectionResults.Add(new Face(BBox.FromRect(box), score));
        }
        
        return detectionResults;
    }
    
    /// <summary>
    /// Applies Non-maximum suppression to a list of faces
    /// </summary>
    /// <param name="faces">List to filter</param>
    /// <param name="iouThreshold">Intersection over union threshold</param>
    /// <returns>Filtered list</returns>
    public static List<Face> NonMaximumSuppression(List<Face> faces, float iouThreshold)
    {
        var pickedFaces = new List<Face>();

        faces.Sort((a, b) => b.Confidence.CompareTo(a.Confidence)); // sort by score
        
        foreach (var face in faces)
        {
            var overlap = pickedFaces.Select(pickedFace => IoU(face, pickedFace)).Any(iou => iou > iouThreshold);

            if (!overlap)
            {
                pickedFaces.Add(face);
            }
        }

        return pickedFaces;
    }
    
    /// <summary>
    /// Intersection over union between two faces
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <returns>Intersection area over union area</returns>
    /// Formula:
    /// x_{IoU} = \frac{A \cap B}{A \cup B}
    /// where A and B are the areas of the two bounding boxes
    /// and A ∩ B is the area of their intersection.
    private static float IoU(Face a, Face b)
    {
        var aBox = a.BBox.ToRect();
        var bBox = b.BBox.ToRect();
        var intersection = SKRect.Intersect(aBox, bBox);
        var union = SKRect.Union(aBox, bBox);
        return intersection.Width * intersection.Height / (union.Width * union.Height);
    }

    public void Dispose()
    {
        _session.Dispose();
    }
}