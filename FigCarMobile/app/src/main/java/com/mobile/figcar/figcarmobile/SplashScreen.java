package com.mobile.figcar.figcarmobile;

import android.content.Intent;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.File;
import java.io.IOException;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;


public class SplashScreen extends AppCompatActivity {
    TextView responseVar;
    Boolean isTrue = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash_screen);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        String path = "/storage/emulated/0/Android/data/com.mobile.figcar.figcarmobile/files/pic.jpg";
        ImageView imageView = (ImageView) findViewById(R.id.imageView);
        imageView.setImageBitmap(BitmapFactory.decodeFile(path));
        responseVar = findViewById(R.id.responseVar);

        SendImage task = new SendImage();
        task.execute();



        ImageButton backButton = findViewById(R.id.backButton);
        backButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                CheckBox feedback = findViewById(R.id.checkBox);
                if (feedback.isChecked()) {
                    Snackbar snackbar = Snackbar.make(v,"Thanks for the feedback. Unfortunately, this won't help with accuracy that much.", Snackbar.LENGTH_SHORT);
                    snackbar.show();
                }

                Intent intent = new Intent(SplashScreen.this, CameraActivity.class);
                startActivity(intent);
            }

        });

        Button send = findViewById(R.id.Send);
        send.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                CheckBox feedback = findViewById(R.id.checkBox);
                if (feedback.isChecked() && isTrue) {
                    Snackbar snackbar = Snackbar.make(v,"Thanks for the feedback. Unfortunately, this won't help with accuracy that much.", Snackbar.LENGTH_SHORT);
                    snackbar.show();
                }
            }

        });



    }

    private class SendImage extends AsyncTask<Void, Void, Void> {
        String res = "No Response";
        long before = 0;
        long after = 0;

        @Override
        protected Void doInBackground(Void... params) {
            before = System.currentTimeMillis();
            String UPLOAD_URL = "http://35.229.36.158:2222/upload";
            String path = "/storage/emulated/0/Android/data/com.mobile.figcar.figcarmobile/files/pic.jpg";

            File image = new File(path);

            OkHttpClient client = new OkHttpClient();

            RequestBody requestBody = new MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("image", image.getName(),
                            RequestBody.create(MediaType.parse("image/jpeg"), image))
                    .build();

            Request request = new Request.Builder()
                    .url(UPLOAD_URL)
                    .post(requestBody)
                    .build();

            Response response = null;

            try {
                response = client.newCall(request).execute();
                res = response.body().string();
                after = System.currentTimeMillis();
                Log.v("Response", res);
                responseVar.setText(new StringBuilder().append(res).append("\n").append("It took ").append((after - before)/1000).append(" second(s).").toString());
                isTrue = true;
            } catch (IOException e) {
                e.printStackTrace();
            }

            if (response == null || !response.isSuccessful()) {
                Log.w("Example", "Unable to upload to server.");
            } else {
                Log.v("Example", "Upload was successful.");
            }
            return null;
        }

        @Override
        protected void onPostExecute(Void result) {
            if (res.contains("success")) {
                responseVar.setText(new StringBuilder().append(res).append("\n").append("It took ").append(1000 * (after - before)).append(" seconds.").toString());
            }
        }
    }

}
