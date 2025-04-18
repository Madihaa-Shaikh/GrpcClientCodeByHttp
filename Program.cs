using System;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.Collections.Generic;
using OpenCvSharp;
using System.ComponentModel.DataAnnotations;

class Program
{
    static async Task Main(string[] args)
    {
        string serverUrl = "http://localhost:8000/v2/models/yolov3/versions/1/infer"; // YOLOv3 model endpoint
        string labelsFilePath = "/mnt/c/Users/Hiwi/source/repos/TritonForPhoto/GrpcClient/imagenet_classes.txt"; // Update with ImageNet or YOLO class labels
        string[] classLabels = File.Exists(labelsFilePath) ? File.ReadAllLines(labelsFilePath) : new string[1000];
        string imagePath = "/mnt/c/Users/Hiwi/source/repos/TritonForPhoto/GrpcClient/test.jpeg"; // Your image path

        var inputData = new
        {
            id = "42",
            inputs = new[]
            {
                new
                {
                    name = "image_shape",  // Name of the input tensor for image dimensions
                    shape = new[] { 1, 2 },  // Shape: [1, 2] for YOLOv3 (height, width)
                    datatype = "FP32",
                    data = new float[] { 416f, 416f } // Size: 416x416 for YOLOv3
                },
                new
                {
                    name = "input_1",  // Name of the input tensor for the actual image
                    shape = new[] { 1, 3, 416, 416 },  // Shape: [1, 3, 416, 416] for YOLOv3
                    datatype = "FP32",
                    data = PreprocessImage(imagePath) // Preprocessed image data
                }
            },
            outputs = new[]
            {
                new
                {
                    name = "yolonms_layer_1/ExpandDims_1:0"  // Update based on your model's output tensor names
                },
                new
                {
                    name = "yolonms_layer_1/ExpandDims_3:0"  // Same as above
                }
            }
        };

        string jsonData = JsonConvert.SerializeObject(inputData);

        using (HttpClient client = new HttpClient())
        {
            var content = new StringContent(jsonData, Encoding.UTF8, "application/json");

            try
            {
                HttpResponseMessage response = await client.PostAsync(serverUrl, content);

                if (response.IsSuccessStatusCode)
                {
                    string responseData = await response.Content.ReadAsStringAsync();
                    var inferenceResponse = JsonConvert.DeserializeObject<dynamic>(responseData);

                    Console.WriteLine("Inference Response:");
                    Console.WriteLine(inferenceResponse);

                    var outputContents = inferenceResponse.outputs[0].data.ToObject<float[]>();  // Extract the output data

                    List<float> outputList = new List<float>(outputContents);

                    // Getting predicted class
                    int predictedClassIndex = outputList.IndexOf(outputList.Max());
                    Console.WriteLine($"Predicted Class Index: {predictedClassIndex}");
                    string predictedClassLabel = predictedClassIndex < classLabels.Length ? classLabels[predictedClassIndex] : "Unknown";
                    Console.WriteLine($"Predicted Class Label: {predictedClassLabel}");
                }
                else
                {
                    Console.WriteLine("Error during inference request.");
                    Console.WriteLine($"Status Code: {response.StatusCode}, Message: {response.ReasonPhrase}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Exception: {ex.Message}");
            }
        }
    }

    private static float[] PreprocessImage(string imagePath)
    {
        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException($"Image not found: {imagePath}");
        }

        using Mat image = Cv2.ImRead(imagePath, ImreadModes.Color);

        if (image.Empty())
        {
            throw new Exception("Failed to load image.");
        }

        // Resize the image to 416x416 (common size for YOLOv3)
        Cv2.Resize(image, image, new OpenCvSharp.Size(416, 416));

        // Convert the image to float and normalize [0, 1]
        image.ConvertTo(image, MatType.CV_32F, 1.0 / 255.0);

        // Normalize based on ImageNet mean and std values (adjust for YOLOv3 if necessary)
        var mean = new Vec3f(0.485f, 0.456f, 0.406f);
        var std = new Vec3f(0.229f, 0.224f, 0.225f);

        for (int y = 0; y < image.Rows; y++)
        {
            for (int x = 0; x < image.Cols; x++)
            {
                Vec3f pixel = image.Get<Vec3f>(y, x);
                pixel.Item0 = (pixel.Item0 - mean.Item0) / std.Item0;
                pixel.Item1 = (pixel.Item1 - mean.Item1) / std.Item1;
                pixel.Item2 = (pixel.Item2 - mean.Item2) / std.Item2;
                image.Set<Vec3f>(y, x, pixel);
            }
        }

        if (image.Channels() != 3)
        {
            throw new Exception("Image must have 3 channels (RGB).");
        }

        float[] tensorData = new float[image.Width * image.Height * image.Channels()];

        for (int y = 0; y < image.Rows; y++)
        {
            for (int x = 0; x < image.Cols; x++)
            {
                Vec3f pixel = image.Get<Vec3f>(y, x);
                int index = (y * image.Cols + x) * 3;
                tensorData[index] = pixel.Item0;
                tensorData[index + 1] = pixel.Item1;
                tensorData[index + 2] = pixel.Item2;
            }
        }

        return tensorData;
    }
}
