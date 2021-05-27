/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.codelabs.textclassification;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import com.google.android.gms.tasks.Continuation;
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions;
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager;
import com.google.firebase.ml.custom.FirebaseCustomRemoteModel;
import java.io.File;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.task.text.nlclassifier.NLClassifier;


public class MainActivity extends AppCompatActivity {
  private static final String TAG = "TextClassificationDemo";

  private TextView resultTextView;
  private EditText inputEditText;
  private ExecutorService executorService;
  private ScrollView scrollView;
  private Button predictButton;
  private NLClassifier textClassifier;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.tfe_tc_activity_main);
    Log.v(TAG, "onCreate");

    executorService = Executors.newSingleThreadExecutor();
    resultTextView = findViewById(R.id.result_text_view);
    inputEditText = findViewById(R.id.input_text);
    scrollView = findViewById(R.id.scroll_view);

    predictButton = findViewById(R.id.predict_button);
    predictButton.setOnClickListener(
        (View v) -> {
          classify(inputEditText.getText().toString());
        });

    downloadModel("sentiment_analysis");
  }


  private void classify(final String text) {
    executorService.execute(
        () -> {

          List<Category> results = textClassifier.classify(text);


          String textToShow = "Input: " + text + "\nOutput:\n";
          for (int i = 0; i < results.size(); i++) {
            Category result = results.get(i);
            textToShow +=
                String.format("    %s: %s\n", result.getLabel(), result.getScore());
          }
          textToShow += "---------\n";

          // Show classification result on screen
          showResult(textToShow);
        });
  }


  private void showResult(final String textToShow) {
    // Run on UI thread as we'll updating our app UI
    runOnUiThread(
        () -> {
          // Append the result to the UI.
          resultTextView.append(textToShow);

          // Clear the input text.
          inputEditText.getText().clear();

          // Scroll to the bottom to show latest entry's classification result.
          scrollView.post(() -> scrollView.fullScroll(View.FOCUS_DOWN));
        });
  }


  private synchronized void downloadModel(String modelName) {
    final FirebaseCustomRemoteModel remoteModel =
        new FirebaseCustomRemoteModel
            .Builder(modelName)
            .build();
    FirebaseModelDownloadConditions conditions =
        new FirebaseModelDownloadConditions.Builder()
            .requireWifi()
            .build();
    final FirebaseModelManager firebaseModelManager = FirebaseModelManager.getInstance();
    firebaseModelManager
        .download(remoteModel, conditions)
        .continueWithTask(task ->
            firebaseModelManager.getLatestModelFile(remoteModel)
        )
        .continueWith(executorService, (Continuation<File, Void>) task -> {
          // Initialize a text classifier instance with the model
          File modelFile = task.getResult();


          textClassifier = NLClassifier.createFromFile(modelFile);

          // Enable predict button
          predictButton.setEnabled(true);
          return null;
        })
        .addOnFailureListener(e -> {
          Log.e(TAG, "Failed to download and initialize the model. ", e);
          Toast.makeText(
              MainActivity.this,
              "Model download failed, please check your connection.",
              Toast.LENGTH_LONG)
              .show();
          predictButton.setEnabled(false);
        });
  }
}
