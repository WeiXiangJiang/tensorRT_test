import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from torchvision import transforms
import numpy as np
from PIL import Image
import time
import argparse
import os
import gc

def normalize(v):
    r = np.asarray(v)
    row, col = v.shape

    for index in range(row):
        line = v[index, ...]
        l2 = np.linalg.norm(line)

        r[index, ...] = v[index, ...] / l2
    # end for row

    return r

def loadEngine2TensorRT(filepath):
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # 反序列化引擎
    with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine

def do_inference(context, batch_size, input, output_shape):

    #context = engine.create_execution_context()
    output = np.empty(output_shape, dtype=np.float32)

    # 分配内存
    d_input = cuda.mem_alloc(1 * input.size * input.dtype.itemsize)
    d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
    bindings = [int(d_input), int(d_output)]

    # pycuda操作缓冲区
    stream = cuda.Stream()
    # 将输入数据放入device
    cuda.memcpy_htod_async(d_input, input, stream)

    start = time.time()
    # 执行模型
    context.execute_async(batch_size, bindings, stream.handle, None)
    # 将预测结果从从缓冲区取出
    cuda.memcpy_dtoh_async(output, d_output, stream)
    end = time.time()

    # 线程同步
    stream.synchronize()
    #output1=normalize(output.reshape((1,512)))
    #sum =0 
    #for i in output.reshape((512)):
    #    sum=sum+i
    #sum=sum**0.5
   # print("sum:",sum)
    #output1=output/sum
    #
    output1=""
    for i in output[0]:
        output1=output1+" "+str(i)
    print("\nTensorRT {} test:".format(engine_path.split('/')[-1].split('.')[0]))
    #print("output:", output)
    #print("output1:",output1)
    print("time cost:", end - start)
    # del d_input
    d_input.free()
    d_output.free()
    # del d_output
    return(output1)

def get_shape(engine):
    for binding in engine:
        if engine.binding_is_input(binding):
            input_shape = engine.get_binding_shape(binding)
        else:
            output_shape = engine.get_binding_shape(binding)
    return input_shape, output_shape

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "TensorRT do inference")
    parser.add_argument("--batch_size", type=int, default=1, help='batch_size')
    parser.add_argument("--repeat", type=int, default='200000', help='test image count')
    parser.add_argument("--engine_file_path", type=str, default='my_files/test.engine', help='engine_file_path')
    args = parser.parse_args()

    engine_path = args.engine_file_path
    engine = loadEngine2TensorRT(engine_path)
    repeat=args.repeat
    """
    img = Image.open(args.img_path)
    input_shape, output_shape = get_shape(engine)
    transform = transforms.Compose([
        transforms.Resize([input_shape[1], input_shape[2]]),  # [h,w]
        transforms.ToTensor()
        ])
    img = transform(img).unsqueeze(0)
    img = img.numpy()
    
    """
    input_shape, output_shape = get_shape(engine)
    files_path = '/home/econe/peter/Pytorch2TensorRT/imagelist.txt'
    txt_file = open(files_path, 'r')
    lines = txt_file.readlines()
    imgs = [os.path.join('/home/econe/peter/Pytorch2TensorRT/data/AT/insightface/thirdparty/',
                              line.rstrip()) for line in lines]
    context = engine.create_execution_context()
    for i in range(len(imgs)):
        if i < repeat:
            print("test image name: %s"%(imgs[i]))
            img = cv2.imread(imgs[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32).transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)
            mean_rgb = [127.5, 127.5, 127.5]
            std_rgb = [127.5, 127.5, 127.5]
            for k in range(3):
                img[0, k, :, :] = (img[0, k, :, :] - mean_rgb[k]) / std_rgb[k]
            img = img.reshape((input_shape[1], input_shape[2], input_shape[3]))
            img=np.ascontiguousarray(img)
            output=do_inference(context, args.batch_size, img, output_shape)
            os.system("echo %s >> result.txt"%(output))
    print("****test finished****")
