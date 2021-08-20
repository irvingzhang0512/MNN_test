/**
TensorFlow Object Detection API 模型转换过程

cd /PATH/TO/MNN/build
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
tar zxvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
./MNNConvert -f TF --modelFile ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb --MNNModel ssd_mobilenet_v1.mnn --bizCode biz
 
**/

#include <MNN/Interpreter.hpp>
using namespace MNN;

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    MNN_PRINT("Usage: ./mnn_pb model.mnn\n");
    return 0;
  }

  // Create session and interpreter
  std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
  ScheduleConfig config;
  config.type = MNN_FORWARD_AUTO;
  auto session = net->createSession(config);
  auto input_tensor = net->getSessionInput(session, NULL);
  // 输出 **Tensor shape**: 1, -1, -1, 3,
  input_tensor->printShape();

  // resize tensor and shape
  auto shape = input_tensor->shape();
  shape[1] = 300;
  shape[2] = 300;
  net->resizeTensor(input_tensor, shape);

  // 这一步报错：段错误 (核心已转储)
  net->resizeSession(session);

  return 0;
}
