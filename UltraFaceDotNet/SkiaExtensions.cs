using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace UltraFaceDotNet;

public static class SkiaExtensions
{
    public static void ExtractPixelsRgbToTensor(this SKBitmap image, DenseTensor<float> tensor)
    {
        var bytesPerPixel = image.BytesPerPixel;
        
        if (bytesPerPixel != 4)
        {
            throw new ArgumentException("Image must be RGBA");
        }
        unsafe // get access to pointers
        {
            var bytes = (byte*)image.GetPixels(); // get pointer to pixels

            for (var y = 0; y < image.Height; y++)
            {
                var row = bytes + (y * image.RowBytes); // get pointer to row
                for (var x = 0; x < image.Width; x++)
                {
                    tensor[0, 0, y, x] = row[x * bytesPerPixel + 2] / 255.0F; // r
                    tensor[0, 1, y, x] = row[x * bytesPerPixel + 1] / 255.0F; // g
                    tensor[0, 2, y, x] = row[x * bytesPerPixel + 0] / 255.0F; // b
                }
            }
        }
    }
    
    public static void ExtractPixelsRgbToTensorSpan(this SKBitmap image, DenseTensor<float> tensor)
    {
        var bytesPerPixel = image.BytesPerPixel;
        
        if (bytesPerPixel != 4)
        {
            throw new ArgumentException("Image must be RGBA");
        }
        var span = image.GetPixelSpan();
        
        for (var y = 0; y < image.Height; y++)
        {
            for (var x = 0; x < image.Width; x++)
            {
                tensor[0, 0, y, x] = span[y * image.Width + x] / 255.0F; // r
                tensor[0, 1, y, x] = span[y * image.Width + x + 1] / 255.0F; // g
                tensor[0, 2, y, x] = span[y * image.Width + x + 2] / 255.0F; // b
            }
        }
        
        
    }
}