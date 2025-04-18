This C# program performs image inference using a YOLOv3 model deployed on a Triton Inference Server. It reads an image from disk, preprocesses it using OpenCV (resize, normalize, and convert to a tensor), and sends the data to the Triton server via HTTP POST in JSON format. The response includes the model's predicted outputs, from which the most probable class is determined and mapped to a human-readable label from a class file (imagenet_classes.txt).

 docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/ubuntu/triton-inference-server/model_repository:/models nvcr.io/nvidia/tritonserver:23.06-py3 tritonserver --model-repository=/models

I0418 11:05:48.625573 1 grpc_server.cc:2445] Started GRPCInferenceService at 0.0.0.0:8001
I0418 11:05:48.625837 1 http_server.cc:3555] Started HTTPService at 0.0.0.0:8000
I0418 11:05:48.670114 1 http_server.cc:185] Started Metrics Service at 0.0.0.0:8002

Output:   0.0,
        8.516497587152116E-07,
        1.0801407768212812E-07,
        0.0,
        6.315707992143871E-07,
        1.1723676607289235E-07,
        2.863842496481084E-10,
        2.663362579369277E-07,
        1.6754766818394273E-07,
        3.4661695735849207E-10,
        3.846774916382856E-07,
        1.1485978745895409E-07,
        2.3164403728515026E-10,
        3.9366457826872647E-07,
        1.7970222643270972E-08,
        4.0898839870351367E-11,
        2.5779016255000897E-07,
        1.4799937275711272E-08,
        7.469225238310173E-11,
        5.271202212497883E-07,
        9.114245358432527E-09,
        4.902744876744691E-11,
        3.0314024002109363E-07,
        1.8356558939558454E-08,
        1.5155876553762937E-10,
        4.183293356163631E-07,
        2.682145350263454E-08,
        9.197265171678737E-11,
        1.543903067613428E-06,
        5.193776075884671E-08,
        1.2751399935950758E-10,
        1.5646446627215482E-06,
        6.242225936148316E-08,
        9.340794804302277E-11,
        5.737313131248811E-07,
        1.392572812619619E-07,
        1.2733778476103907E-09,
        2.441656477003562E-07,
        3.6048072615813E-07,
        1.1987602022145438E-08,
        1.738870025747019E-08
      ]
    }
  ]
}
Predicted Class Index: 243
Predicted Class Label: bull mastiff



I0418 11:25:44.058428 1 server.cc:662]
+--------------------+---------+--------+
| Model              | Version | Status |
+--------------------+---------+--------+
| densenet_onnx      | 1       | READY  |
| inception_graphdef | 1       | READY  |
| yolov3             | 1       | READY  |
+--------------------+---------+--------+
