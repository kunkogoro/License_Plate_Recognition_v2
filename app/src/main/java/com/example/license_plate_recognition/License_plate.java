package com.example.license_plate_recognition;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.text.format.Formatter;
import android.util.Base64;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.SpinnerAdapter;
import android.widget.TextView;
import android.widget.Toast;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.gun0912.tedpermission.PermissionListener;
import com.gun0912.tedpermission.normal.TedPermission;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import gun0912.tedbottompicker.TedBottomPicker;

public class License_plate extends AppCompatActivity {

    private Button btnLoad,btnProsse;
    private ImageView imv;
    private TextView tv;
    private BitmapDrawable bitmapDrawable;
    private Bitmap bitmap;
    private String imageString = "";
    private Spinner spinner;
    private List<String> list_mode;
    private String mode;
    private String URI;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_license_plate);

        btnLoad = findViewById(R.id.button);
        btnProsse = findViewById(R.id.button1);
        imv = findViewById(R.id.imageView);
        tv = findViewById(R.id.text_license);
        spinner = findViewById(R.id.spin_mode);
        list_mode = new ArrayList<>();
        list_mode.add("CNN");
        list_mode.add("SVM");
        list_mode.add("ESRGAN");

        ArrayAdapter arrayAdapter = new ArrayAdapter(this, android.R.layout.simple_spinner_item,list_mode);
        arrayAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(arrayAdapter);

        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                mode = list_mode.get(position);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {

            }
        });

        btnLoad.setOnClickListener(v -> {
            requestPermissions();
        });

        if(!Python.isStarted()){
            Python.start(new AndroidPlatform(this));
        }

        final Python py = Python.getInstance();

        btnProsse.setOnClickListener(v -> {
            bitmapDrawable = (BitmapDrawable) imv.getDrawable();
            bitmap = bitmapDrawable.getBitmap();
            imageString = getStringImage(bitmap);

            PyObject pyObject;

            if (mode.equals("SVM")){
                pyObject = py.getModule("run2");
            }else if(mode.equals("CNN")){
                pyObject = py.getModule("run1");
            }else{
                pyObject = py.getModule("ESGAN");
            }

            PyObject pyObject1 = pyObject.callAttr("processing",URI);


            if (mode.equals("SVM") || mode.equals("CNN")) {
                String deString = pyObject1.toString();
                tv.setText(deString);
            }else{
                String deString = pyObject1.toString();
            byte data[] = Base64.decode(deString,Base64.DEFAULT);
            Bitmap bitmap1 = BitmapFactory.decodeByteArray(data,0,data.length);
            imv.setImageBitmap(bitmap1);
                tv.setText("ok");
            }


        });
    }
    public void requestPermissions(){
        PermissionListener permissionlistener = new PermissionListener() {
            @Override
            public void onPermissionGranted() {
                Toast.makeText(License_plate.this, "Permission Granted", Toast.LENGTH_SHORT).show();
                openImage();
            }

            @Override
            public void onPermissionDenied(List<String> deniedPermissions) {
                Toast.makeText(License_plate.this, "Permission Denied\n" + deniedPermissions.toString(), Toast.LENGTH_SHORT).show();
            }
        };
        TedPermission.create()
                .setPermissionListener(permissionlistener)
                .setDeniedMessage("If you reject permission,you can not use this service\n\nPlease turn on permissions at [Setting] > [Permission]")
                .setPermissions(Manifest.permission.CAMERA, Manifest.permission.READ_EXTERNAL_STORAGE,Manifest.permission.WRITE_EXTERNAL_STORAGE)
                .check();

    }
    public void openImage(){
        TedBottomPicker.OnImageSelectedListener listener = new TedBottomPicker.OnImageSelectedListener(){

            @Override
            public void onImageSelected(Uri uri) {
                try {
//                   Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(),uri);
                    final InputStream imageStream = getContentResolver().openInputStream(uri);

                    final Bitmap bitmap = BitmapFactory.decodeStream(imageStream);
                    imv.setImageBitmap(bitmap);
                    URI = uri.getPath();
                    System.out.println("URI:" + URI);
                  // "/mnt/sdcard/FileName.mp3"
                  //  File file = new File(new URI(path));
//                    originalImage = new File(uri.getPath().replace("raw/",""));
//                    size_image_start.setText("Size: " + Formatter.formatShortFileSize(ImageCompressorPage.this, originalImage.length()));
//                    card_size_start.setVisibility(View.VISIBLE);
                }catch (Exception e){
                    Toast.makeText(License_plate.this, e.getMessage(), Toast.LENGTH_SHORT).show();
                }
            }
        };

        TedBottomPicker tedBottomPicker = new TedBottomPicker.Builder(License_plate.this)
                .setOnImageSelectedListener(listener).create();
        tedBottomPicker.show(getSupportFragmentManager());
    }



    private String getStringImage(Bitmap bitmap) {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG,100,byteArrayOutputStream);
        byte[] bytes = byteArrayOutputStream.toByteArray();
        String encodeImage = Base64.encodeToString(bytes,Base64.DEFAULT);
        return encodeImage;
    }
}