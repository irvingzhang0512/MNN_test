#include <MNN/Interpreter.hpp>
using namespace MNN;

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    MNN_PRINT("Usage: ./yolox.out model.mnn\n");
    return 0;
  }

  // Create session and interpreter
  std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
  ScheduleConfig config;
  config.type = MNN_FORWARD_AUTO;
  auto session = net->createSession(config);

  // create input
  auto input = net->getSessionInput(session, NULL);
  std::shared_ptr<Tensor> inputUser(new Tensor(input, MNN::Tensor::CAFFE));
  auto bpp = inputUser->channel();
  auto size_h = inputUser->height();
  auto size_w = inputUser->width();
  MNN_PRINT("input: w:%d , h:%d, bpp: %d\n", size_w, size_h, bpp);
  int length = bpp * size_h * size_w;
  for (int i = 0; i < length; i++) inputUser->host<float>()[i] = 1.f;
  input->copyFromHostTensor(inputUser.get());

  // inference
  net->runSession(session);

  // get output
  auto output = net->getSessionOutput(session, NULL);
  // std::shared_ptr<Tensor> outputUser(
  //     new Tensor(output, MNN::Tensor::TENSORFLOW));
  std::shared_ptr<Tensor> outputUser(new Tensor(output, MNN::Tensor::CAFFE));
  output->copyToHostTensor(outputUser.get());
  for (int i = 0; i < 10; i++) {
    // 之前这一步写错了，写成了 device tensor
    MNN_PRINT("output: %f\n", outputUser->host<float>()[i]);
  }
  return 0;
}
