using SkiaSharp;

namespace UltraFaceDotNet;

public readonly record struct BBox(float Left, float Top, float Width, float Height)
{
    public SKRectI ToRectI()
    {
        var left = (int) Math.Round(Left, 0);
        var top = (int) Math.Round(Top, 0);
        var right = (int) Math.Round(Left + Width, 0);
        var bottom = (int) Math.Round(Top + Height, 0);
        
        return new SKRectI(left, top, right, bottom);
    }

    public SKRect ToRect()
    {
        var left = Left;
        var top = Top;
        var right = Left + Width;
        var bottom = Top + Height;
        
        return new SKRect(left, top, right, bottom);
    }
    
    public static BBox FromRect(SKRect rect) => new(rect.Left, rect.Top, rect.Width, rect.Height);

    public static BBox FromRectI(SKRectI rect) => new(rect.Left, rect.Top, rect.Width, rect.Height);
}