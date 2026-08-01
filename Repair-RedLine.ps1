param(
    [string]$InputDir = ".",
    [string]$OutputDir = "fix photo",
    [int]$HalfWidth = 7,
    [switch]$AnalyzeOnly
)

Add-Type -AssemblyName System.Drawing

$source = @"
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;

public static class RedLineRepair
{
    public sealed class Result
    {
        public string FileName;
        public string OutputPath;
        public int Width;
        public int Height;
        public int LineX;
        public double Score;
        public int ChangedPixels;
        public int RepairColumns;
    }

    public static Result Process(string inputPath, string outputDir, int halfWidth, bool analyzeOnly)
    {
        using (Image loaded = Image.FromFile(inputPath))
        using (Bitmap bitmap = new Bitmap(loaded.Width, loaded.Height, PixelFormat.Format24bppRgb))
        {
            using (Graphics graphics = Graphics.FromImage(bitmap))
            {
                graphics.DrawImage(loaded, 0, 0, loaded.Width, loaded.Height);
            }

            Rectangle rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            BitmapData data = bitmap.LockBits(rect, ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            int stride = data.Stride;
            int byteCount = Math.Abs(stride) * bitmap.Height;
            byte[] pixels = new byte[byteCount];
            Marshal.Copy(data.Scan0, pixels, 0, byteCount);

            int lineX = DetectLineColumn(pixels, bitmap.Width, bitmap.Height, stride, out double score);
            int changed = 0;
            int repairColumns = Math.Max(1, halfWidth * 2 + 1);

            if (!analyzeOnly)
            {
                changed = RepairStripe(pixels, bitmap.Width, bitmap.Height, stride, lineX, halfWidth);
                Marshal.Copy(pixels, 0, data.Scan0, byteCount);
            }

            bitmap.UnlockBits(data);

            string outPath = "";
            if (!analyzeOnly)
            {
                Directory.CreateDirectory(outputDir);
                string baseName = Path.GetFileNameWithoutExtension(inputPath);
                outPath = Path.Combine(outputDir, baseName + "_fixed.png");
                bitmap.Save(outPath, ImageFormat.Png);
            }

            return new Result
            {
                FileName = Path.GetFileName(inputPath),
                OutputPath = outPath,
                Width = bitmap.Width,
                Height = bitmap.Height,
                LineX = lineX,
                Score = score,
                ChangedPixels = changed,
                RepairColumns = repairColumns
            };
        }
    }

    private static int DetectLineColumn(byte[] pixels, int width, int height, int stride, out double bestScore)
    {
        int bestX = width / 2;
        bestScore = Double.MinValue;

        for (int x = 3; x < width - 3; x++)
        {
            double score = 0.0;

            for (int y = 0; y < height; y++)
            {
                int row = y * stride;
                int center = row + x * 3;
                int left = row + (x - 2) * 3;
                int right = row + (x + 2) * 3;

                int b = pixels[center + 0];
                int g = pixels[center + 1];
                int r = pixels[center + 2];

                int ib = (pixels[left + 0] + pixels[right + 0]) / 2;
                int ig = (pixels[left + 1] + pixels[right + 1]) / 2;
                int ir = (pixels[left + 2] + pixels[right + 2]) / 2;

                int dr = r - ir;
                int dg = g - ig;
                int db = b - ib;
                int redExcess = dr - ((dg + db) / 2);
                int localDistance = Math.Abs(dr) + Math.Abs(dg) + Math.Abs(db);

                if (redExcess > 4)
                {
                    score += (redExcess - 4);
                }

                if (localDistance > 18 && dr > 0)
                {
                    score += (localDistance - 18) * 0.15;
                }
            }

            if (score > bestScore)
            {
                bestScore = score;
                bestX = x;
            }
        }

        return bestX;
    }

    private static int RepairStripe(byte[] pixels, int width, int height, int stride, int centerX, int halfWidth)
    {
        int startX = Math.Max(1, centerX - halfWidth);
        int endX = Math.Min(width - 2, centerX + halfWidth);
        int columnCount = endX - startX + 1;
        int changed = 0;

        for (int y = 0; y < height; y++)
        {
            int row = y * stride;
            int left = row + (startX - 1) * 3;
            int right = row + (endX + 1) * 3;

            int lb = pixels[left + 0];
            int lg = pixels[left + 1];
            int lr = pixels[left + 2];
            int rb = pixels[right + 0];
            int rg = pixels[right + 1];
            int rr = pixels[right + 2];

            for (int x = startX; x <= endX; x++)
            {
                int center = row + x * 3;
                double t = (double)(x - startX + 1) / (double)(columnCount + 1);

                byte nb = (byte)Math.Round(lb + ((rb - lb) * t));
                byte ng = (byte)Math.Round(lg + ((rg - lg) * t));
                byte nr = (byte)Math.Round(lr + ((rr - lr) * t));

                if (pixels[center + 0] != nb || pixels[center + 1] != ng || pixels[center + 2] != nr)
                {
                    changed++;
                    pixels[center + 0] = nb;
                    pixels[center + 1] = ng;
                    pixels[center + 2] = nr;
                }
            }
        }

        return changed;
    }
}
"@

$references = @(
    (Join-Path $PSHOME "System.Runtime.dll"),
    (Join-Path $PSHOME "System.Runtime.InteropServices.dll"),
    (Join-Path $PSHOME "System.Drawing.dll"),
    (Join-Path $PSHOME "System.Drawing.Common.dll"),
    (Join-Path $PSHOME "System.Drawing.Primitives.dll"),
    (Join-Path $PSHOME "System.Private.Windows.Core.dll"),
    (Join-Path $PSHOME "System.Private.Windows.GdiPlus.dll")
) | Where-Object { Test-Path -LiteralPath $_ }

Add-Type -TypeDefinition $source -ReferencedAssemblies $references

$inputPath = Resolve-Path -LiteralPath $InputDir
$outputPath = Join-Path $inputPath $OutputDir
$images = Get-ChildItem -LiteralPath $inputPath -File |
    Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|webp|bmp|tif|tiff)$' }

foreach ($image in $images) {
    $result = [RedLineRepair]::Process($image.FullName, $outputPath, $HalfWidth, [bool]$AnalyzeOnly)
    [pscustomobject]@{
        File = $result.FileName
        Size = "{0}x{1}" -f $result.Width, $result.Height
        LineX = $result.LineX
        Score = [math]::Round($result.Score, 2)
        RepairColumns = $result.RepairColumns
        ChangedPixels = $result.ChangedPixels
        Output = $result.OutputPath
    }
}
