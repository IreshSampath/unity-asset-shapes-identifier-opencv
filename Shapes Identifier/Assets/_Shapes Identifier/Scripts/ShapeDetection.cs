using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using System.Collections.Generic;
using UnityEngine;
using Rect = OpenCVForUnity.CoreModule.Rect;

public class ShapeDetection : MonoBehaviour
{
    public Texture2D _inputTexture;
    public Renderer _outputTexture;

    public List<Texture2D> _predefinedTextures;

    public double _threshold = 0.8;

    void Start()
    {
        CheckFullImage();
        //CheckTopPartOfImage();
        //CheckBottomPartOfImage();
    }

    public void CheckFullImage()
    {
        // Convert user-assigned image to OpenCV Mat
        Mat inputMat = new Mat(_inputTexture.height, _inputTexture.width, CvType.CV_8UC4);
        Utils.texture2DToMat(_inputTexture, inputMat);

        // Convert user-assigned image to grayscale
        Mat inputGray = new Mat();
        Imgproc.cvtColor(inputMat, inputGray, Imgproc.COLOR_RGBA2GRAY);

        // Compare with each predefined image
        foreach (var predefinedTexture in _predefinedTextures)
        {
            // Convert predefined image to OpenCV Mat
            Mat predefinedMat = new Mat(predefinedTexture.height, predefinedTexture.width, CvType.CV_8UC4);
            Utils.texture2DToMat(predefinedTexture, predefinedMat);

            // Convert predefined image to grayscale
            Mat predefinedGray = new Mat();
            Imgproc.cvtColor(predefinedMat, predefinedGray, Imgproc.COLOR_RGBA2GRAY);

            // Perform template matching
            Mat result = new Mat();
            Imgproc.matchTemplate(inputGray, predefinedGray, result, Imgproc.TM_CCOEFF_NORMED);

            // Find the best match
            Core.MinMaxLocResult matchResult = Core.minMaxLoc(result);
            double maxVal = matchResult.maxVal; // Confidence value (0 to 1)

            // If the match is above a _threshold, consider it a match
            if (maxVal > _threshold)
            {
                Debug.Log("Match found with confidence: " + maxVal);
                // Optionally, draw a rectangle around the matched area
                DrawMatchRectangle(inputMat, predefinedMat, matchResult);
            }
            else
            {
                Debug.Log("No match found.");
            }
        }

        // Display the result
        Texture2D outputTexture = new Texture2D(inputMat.cols(), inputMat.rows(), TextureFormat.RGBA32, false);
        Utils.matToTexture2D(inputMat, outputTexture);
        _outputTexture.material.mainTexture = outputTexture;
    }

    public void CheckTopPartOfImage()
    {
        // Convert user-assigned image to OpenCV Mat
        Mat inputMat = new Mat(_inputTexture.height, _inputTexture.width, CvType.CV_8UC4);
        Utils.texture2DToMat(_inputTexture, inputMat);

        // Crop the top half of the input image
        Rect topHalfRect = new Rect(0, 0, inputMat.cols(), inputMat.rows() / 2);
        Mat topHalfMat = new Mat(inputMat, topHalfRect);

        // Convert the top half to grayscale
        Mat topHalfGray = new Mat();
        Imgproc.cvtColor(topHalfMat, topHalfGray, Imgproc.COLOR_RGBA2GRAY);

        // Compare with each predefined image
        foreach (var predefinedTexture in _predefinedTextures)
        {
            // Convert predefined image to OpenCV Mat
            Mat predefinedMat = new Mat(predefinedTexture.height, predefinedTexture.width, CvType.CV_8UC4);
            Utils.texture2DToMat(predefinedTexture, predefinedMat);

            // Convert predefined image to grayscale
            Mat predefinedGray = new Mat();
            Imgproc.cvtColor(predefinedMat, predefinedGray, Imgproc.COLOR_RGBA2GRAY);

            // Perform template matching
            Mat result = new Mat();
            Imgproc.matchTemplate(topHalfGray, predefinedGray, result, Imgproc.TM_CCOEFF_NORMED);

            // Find the best match
            Core.MinMaxLocResult matchResult = Core.minMaxLoc(result);
            double maxVal = matchResult.maxVal; // Confidence value (0 to 1)

            if (maxVal > _threshold)
            {
                Debug.Log("Match found with confidence: " + maxVal);
                // Optionally, draw a rectangle around the matched area
                DrawMatchRectangle(inputMat, predefinedMat, matchResult, "top");
            }
            else
            {
                Debug.Log("No match found.");
            }
        }

        // Display the result
        Texture2D outputTexture = new Texture2D(inputMat.cols(), inputMat.rows(), TextureFormat.RGBA32, false);
        Utils.matToTexture2D(inputMat, outputTexture);
        _outputTexture.material.mainTexture = outputTexture;
    }

    public void CheckBottomPartOfImage()
    {
        // Convert user-assigned image to OpenCV Mat
        Mat inputMat = new Mat(_inputTexture.height, _inputTexture.width, CvType.CV_8UC4);
        Utils.texture2DToMat(_inputTexture, inputMat);

        // Crop the bottom half of the input image
        Rect bottomHalfRect = new Rect(0, inputMat.rows() / 2, inputMat.cols(), inputMat.rows() / 2);
        Mat bottomHalfMat = new Mat(inputMat, bottomHalfRect);

        // Convert the bottom half to grayscale
        Mat bottomHalfGray = new Mat();
        Imgproc.cvtColor(bottomHalfMat, bottomHalfGray, Imgproc.COLOR_RGBA2GRAY);

        // Compare with each predefined image
        foreach (var predefinedTexture in _predefinedTextures)
        {
            // Convert predefined image to OpenCV Mat
            Mat predefinedMat = new Mat(predefinedTexture.height, predefinedTexture.width, CvType.CV_8UC4);
            Utils.texture2DToMat(predefinedTexture, predefinedMat);

            // Convert predefined image to grayscale
            Mat predefinedGray = new Mat();
            Imgproc.cvtColor(predefinedMat, predefinedGray, Imgproc.COLOR_RGBA2GRAY);

            // Perform template matching
            Mat result = new Mat();
            Imgproc.matchTemplate(bottomHalfGray, predefinedGray, result, Imgproc.TM_CCOEFF_NORMED);

            // Find the best match
            Core.MinMaxLocResult matchResult = Core.minMaxLoc(result);
            double maxVal = matchResult.maxVal; // Confidence value (0 to 1)

            if (maxVal > _threshold)
            {
                Debug.Log("Match found with confidence: " + maxVal);
                // Optionally, draw a rectangle around the matched area
                DrawMatchRectangle(inputMat, predefinedMat, matchResult, "bottom");
            }
            else
            {
                Debug.Log("No match found.");
            }
        }

        // Display the result
        Texture2D outputTexture = new Texture2D(inputMat.cols(), inputMat.rows(), TextureFormat.RGBA32, false);
        Utils.matToTexture2D(inputMat, outputTexture);
        _outputTexture.material.mainTexture = outputTexture;
    }


    //private void DrawMatchRectangle(Mat inputMat, Mat templateMat, Core.MinMaxLocResult matchResult)
    //{
    //    // Calculate the position of the match in the bottom half
    //    Point matchLoc = new Point(matchResult.maxLoc.x, matchResult.maxLoc.y + inputMat.rows() / 2);

    //    // Draw a rectangle around the matched area
    //    Imgproc.rectangle(inputMat, matchLoc, new Point(matchLoc.x + templateMat.cols(), matchLoc.y + templateMat.rows()), new Scalar(0, 255, 0), 2);
    //}


    // Helper function to draw a rectangle around the matched area
    void DrawMatchRectangle(Mat inputMat, Mat templateMat, Core.MinMaxLocResult matchResult, string region = "full")
    {
        //Point matchLoc = matchResult.maxLoc;
       // Point topLeft = matchLoc;
        //Point topLeft = new Point(matchLoc.x, matchLoc.y);

        // Adjust match location based on the selected region
        switch (region.ToLower())
        {
            case "top":
                Point matchLoc = new Point(matchResult.maxLoc.x,  inputMat.rows()/2);
                Point bottomRight = new Point(matchLoc.x + templateMat.cols(), matchLoc.y + templateMat.rows());
                Imgproc.rectangle(inputMat, matchLoc, bottomRight, new Scalar(0, 255, 0), 2);
                break;
            case "bottom":
                //matchLoc = new Point(matchResult.maxLoc.x, matchResult.maxLoc.y + inputMat.rows() / 2);
                //Point bottomRight = new Point(matchLoc.x + templateMat.cols(), matchLoc.y + templateMat.rows());
                //Imgproc.rectangle(inputMat, topLeft, bottomRight, new Scalar(0, 255, 0), 2);
                break;
            case "full":
                break;
            default:
                Debug.LogWarning("Invalid region specified. Using 'full' as default.");
                break;
        }

        // Draw the rectangle
        //Point bottomRight = new Point(matchLoc.x + templateMat.cols(), matchLoc.y + templateMat.rows());
        //Imgproc.rectangle(inputMat, topLeft, bottomRight, new Scalar(0, 255, 0), 2);
    }
}

