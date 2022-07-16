package com.tencent.tnn.demo;

import android.graphics.Bitmap;
import android.util.Log;

public class TNNLib {

    public static final String LIB_NAME_TNN = "tnn_wrapper";

    private long nativePtr;

    static {
        try {
            System.loadLibrary(com.tencent.tnn.demo.TNNLib.LIB_NAME_TNN);
        }catch(Exception e) {
            e.printStackTrace();
        }catch(Error e) {
            e.printStackTrace();
        } finally {
        }
    }

    public void setNativePtr(long nativePtr) {
        this.nativePtr = nativePtr;
    }

    public long getNativePtr(){
        return nativePtr;
    }


    public TNNLib() {}

    public native int init(String protoFilePath, String modelFilePath, String device_type);

    public native float[] forward(Bitmap imageSrc);

    public native int deinit();

}
