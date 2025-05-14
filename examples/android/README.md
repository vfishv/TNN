


echo "alias python=/usr/bin/python3" >> ~/.zshrc
source ~/.zshrc


set(TNN_OPENCL_ENABLE OFF CACHE BOOL "" FORCE)

minSdkVersion 21


``` 

cd <path-to-tnn>/tools/onnx2tnn/onnx-converter
./build.sh

```



``` 

cd ../tools/onnx2tnn/onnx-converter
./build.sh
python3 onnx2tnn.py -h

```

```
python3 -m venv path/to/venv
source path/to/venv/bin/activate
python3 -m pip install onnx
```



