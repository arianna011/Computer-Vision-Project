package com.example.remotebootlegscore

import android.Manifest
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Base64
import android.util.Log
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import okhttp3.MultipartBody

import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream


class MainActivity : ComponentActivity() {

    // server connection

    //private val apiService by lazy { RetrofitClient.createRetroFit(this) }
    private lateinit var apiService: ApiService

    private var ip: String = "192.168.196.228"
    private var port: String = "6969"


    // main view

    private lateinit var imageView: ImageView
    private var currentBitmap: Bitmap? = null
    private var photoUri: Uri? = null

    private var errorTextView: TextView? = null

    // player view

    private lateinit var playerManager: PlayerManager

    private lateinit var playPauseBtn: ImageView
    private lateinit var downloadMidiBtn: ImageButton
    private lateinit var seekBar: SeekBar

    private var author: String = "???"
    private var title: String = "???"

    private var isUserSeeking = false

    // response view

    private var selectedMidiUri: Uri? = null
    private var selectedPdfUri: Uri? = null

    // loading view

    private lateinit var loadingOverlay: View
    private lateinit var loadingSpinner: ProgressBar


    @RequiresApi(Build.VERSION_CODES.Q)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        apiService = RetrofitClient.createRetroFit(this)

        // permissions for camera
        requestPermissions(arrayOf(Manifest.permission.CAMERA), 100)

        // display main screen
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.captured_img)
        imageView.scaleType = ImageView.ScaleType.FIT_CENTER

        errorTextView = findViewById(R.id.error)

        loadingOverlay = findViewById(R.id.loading_overlay)
        loadingSpinner = findViewById(R.id.loading_spinner)

        playPauseBtn = findViewById(R.id.play_pause_button)
        seekBar = findViewById(R.id.seek_bar)
        downloadMidiBtn = findViewById(R.id.download_button_midi)

        // set player manager
        playerManager = PlayerManager(this, { isPlaying ->
            updatePlayPauseButton(isPlaying)
        }, { progress, duration ->
            updateSeekBar(progress, duration)
        })

        // > button handling

        // server selection
        findViewById<Button>(R.id.serverip).setOnClickListener { selectingServer() }

        // image selection
        findViewById<ImageButton>(R.id.capture_button).setOnClickListener { openCamera() }
        findViewById<ImageButton>(R.id.gallery_button).setOnClickListener {
            galleryLauncher.launch(
                "image/*"
            )
        }
        findViewById<ImageButton>(R.id.midi_button).setOnClickListener { openDownloadFolder() }

        // image editing
        findViewById<ImageButton>(R.id.home_button).setOnClickListener { returnHome() }
        findViewById<ImageButton>(R.id.crop_button).setOnClickListener { croppingFeature() }

        // server interaction buttons
        findViewById<Button>(R.id.send_to_server_button_midi).setOnClickListener {
            currentBitmap?.let { it1 ->
                ImageManager.convertBitmapToByteArray(it1)
            }?.let { it2 -> uploadToPythonServer(it2, 0) }
        }
        findViewById<Button>(R.id.send_to_server_button_pdf).setOnClickListener {
            currentBitmap?.let { it1 ->
                ImageManager.convertBitmapToByteArray(it1)
            }?.let { it2 -> uploadToPythonServer(it2, 1) }
        }

        // player bar buttons
        playPauseBtn.setOnClickListener {
            selectedMidiUri?.let {
                if (playerManager.isPlaying())
                    playerManager.pause()
                else
                    playerManager.play(it)
            }
        }
        seekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(
                seekBar: SeekBar?,
                progress: Int,
                fromUser: Boolean
            ) {
                if (fromUser)
                    playerManager.seekTo(progress)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {
                isUserSeeking = true
            }

            override fun onStopTrackingTouch(seekBar: SeekBar?) {
                isUserSeeking = false
            }
        })
        downloadMidiBtn.setOnClickListener {
            selectedMidiUri?.let { uri -> downloadMidiToDevice(uri) }
        }
        findViewById<ImageView>(R.id.download_button_pdf).setOnClickListener {
            selectedPdfUri?.let { uri -> downloadPdfToDevice(uri) }
        }
        findViewById<Button>(R.id.open_pdf).setOnClickListener {
            visualizePdf(
                File(
                    cacheDir,
                    "${author.replace(" ", "_")}_${title.replace(" ", "_")}.pdf"
                )
            )
        }
    }


    // ------ CAMERA FUNCTIONS

    private fun openCamera() {
        val photoFile = ImageManager.createImageFile(this)

        photoFile?.let {
            photoUri = FileProvider.getUriForFile(this, "${packageName}.fileprovider", it)
            val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri)

            if (cameraIntent.resolveActivity(packageManager) != null)
                cameraLauncher.launch(cameraIntent)
            else {
                setImageView(null)
                setErrorStr(getString(R.string.error_camera))
            }

        }
    }


    // ------ SCREEN HANDLING

    private fun setImageView(imgData: ByteArray?) {

        imageView.setImageBitmap(null)
        setPlayerBarVisibility(View.GONE)
        setPdfBarVisibility(View.GONE)

        if (imgData == null) {
            setPhotoBtnVisibility(View.GONE)
            return
        }

        try {
            // set the home view to invisible
            setHomeVisibility(View.GONE)

            // Convert byte array to Bitmap
            val bitmap = BitmapFactory.decodeByteArray(imgData, 0, imgData.size)

            // Show the image in the ImageView
            imageView.setImageBitmap(bitmap)
            imageView.visibility = View.VISIBLE
            currentBitmap = bitmap

            // add extra buttons
            setPhotoBtnVisibility(View.VISIBLE)


            // Hide error message if the image is valid
            errorTextView?.visibility = View.GONE

        } catch (e: Exception) {
            Log.e("IMAGE_ERROR", "Error processing image: ${e.message}")
            errorTextView?.visibility = View.VISIBLE
        }
    }


    private fun returnHome() {

        errorTextView?.visibility = View.GONE
        setPlayerBarVisibility(View.GONE)
        setPdfBarVisibility(View.GONE)
        setHomeVisibility(View.VISIBLE)
        setImageView(null)

    }


    private fun setErrorStr(error: String) {

        setPdfBarVisibility(View.GONE)
        setPlayerBarVisibility(View.GONE)

        errorTextView?.visibility = TextView.VISIBLE
        errorTextView?.text = error

    }


    private fun calculateInSampleSize(options: BitmapFactory.Options, reqWidth: Int, reqHeight: Int): Int {
        val (height: Int, width: Int) = options.outHeight to options.outWidth
        var inSampleSize = 1

        if (height > reqHeight || width > reqWidth) {
            val halfHeight: Int = height / 2
            val halfWidth: Int = width / 2

            // Calculate the largest inSampleSize value that is a power of 2
            // and keeps both dimensions larger than the requested height and width.
            while (halfHeight / inSampleSize >= reqHeight && halfWidth / inSampleSize >= reqWidth) {
                inSampleSize *= 2
            }
        }

        return inSampleSize
    }


    // ------ IMAGE HANDLING

    private fun saveBitmapToFile(bitmap: Bitmap): Uri {
        // Create a temporary file
        val file = File(cacheDir, "temp_image.jpg")
        FileOutputStream(file).use { outStream ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outStream)
        }
        return FileProvider.getUriForFile(this, "com.example.remotebootlegscore.fileprovider", file)
    }


    private fun croppingFeature() {

        currentBitmap?.let { bitmap ->
            val bitmapUri = saveBitmapToFile(bitmap) // Save bitmap to a file and get its URI
            val intent = Intent(this, CroppingActivity::class.java).apply {
                putExtra("image_uri", bitmapUri.toString()) // Pass the URI to CroppingActivity
            }
            croppingLauncher.launch(intent)
        } ?: run {
            Log.e("CROP_ERROR", "No bitmap available to crop")
            setErrorStr(getString(R.string.error_cropping))
        }

    }


    // ------ MIDI MANAGER

    private fun updatePlayPauseButton(isPlaying: Boolean) {
        playPauseBtn.setBackgroundResource(if (isPlaying) R.drawable.pause_png else R.drawable.play_png)
    }


    private fun updateSeekBar(progress: Int, duration: Int) {
        seekBar.max = duration
        seekBar.progress = progress
    }


    private fun saveMidiFile(midiBytes: ByteArray): File? {

        return try {
            val fileName = "${author.replace(" ", "_")}_${title.replace(" ", "_")}.mid"
            val midiFile = File(cacheDir, fileName)
            FileOutputStream(midiFile).use { it.write(midiBytes) }
            midiFile
        } catch (e: Exception) {
            Log.e("MIDI_SAVE_ERROR", "Error saving MIDI file: ${e.message}")
            null
        }
    }


    private fun playMidi(uri: Uri) {
        setPlayerBarVisibility(View.VISIBLE)  // Show player bar
        playerManager.play(uri, forceChange = true)  // Start MIDI playback
    }


    @RequiresApi(Build.VERSION_CODES.Q)
    private fun downloadMidiToDevice(midiUri: Uri) {
        try {
            val midiFile = File(midiUri.path ?: return)
            if (!midiFile.exists()) {
                Log.e("MIDI_DOWNLOAD", "MIDI file not found.")
                return
            }

            val downloadsDir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "Image2Midi")
            if (!downloadsDir.exists()) downloadsDir.mkdirs() // Ensure directory exists

            val fileName = "${author.replace(" ", "_")}_${title.replace(" ", "_")}.mid"
            val outputFile = File(downloadsDir, fileName) // **Explicit .mid extension**

            FileInputStream(midiFile).use { inputStream ->
                FileOutputStream(outputFile).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }

            Log.d("MIDI_DOWNLOAD", "MIDI file saved successfully to: ${outputFile.absolutePath}")

            Toast.makeText(this, "MIDI downloaded", Toast.LENGTH_SHORT).show()


            // Notify media scanner so file appears in file manager
            MediaScannerConnection.scanFile(
                applicationContext,
                arrayOf(outputFile.absolutePath),
                arrayOf("audio/midi"),
                null
            )

        } catch (e: Exception) {
            Log.e("MIDI_DOWNLOAD_ERROR", "Error during MIDI download: ${e.message}")
        }
    }


    // ------ MISC

    private fun openDownloadFolder() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE)
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)

        try {
            startActivityForResult(intent, 200) // Request code for handling result
        } catch (e: Exception) {
            Log.e("OPEN_FOLDER_ERROR", "Error opening folder picker: ${e.message}")
            setErrorStr("Could not open folder picker")
        }
    }


    // ------ PDF MANAGEMENT

    private fun visualizePdf(pdf: File) {
        val pdfUri: Uri = FileProvider.getUriForFile(
            this,
            "${this.packageName}.fileprovider",
            pdf
        )

        val intent = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(pdfUri, "application/pdf")
            flags = Intent.FLAG_GRANT_READ_URI_PERMISSION // Required for sharing
        }

        val chooser = Intent.createChooser(intent, "Open PDF with:")
        this.startActivity(chooser)
    }


    private fun savePdfFile(pdfBytes: ByteArray): File? {
        return try {
            val fileName = "${author.replace(" ", "_")}_${title.replace(" ", "_")}.pdf"
            Log.d("filename", fileName)
            val pdfFile = File(cacheDir, fileName)
            FileOutputStream(pdfFile).use { it.write(pdfBytes) }
            pdfFile
        } catch (e: Exception) {
            Log.e("PDF_SAVE_ERROR", "Error saving PDF file: ${e.message}")
            null
        }
    }


    private fun downloadPdfToDevice(uri: Uri) {
        try {
            val pdfFile = File(uri.path ?: return)
            if (!pdfFile.exists()) {
                Log.e("PDF_DOWNLOAD", "PDF file not found.")
                return
            }

            val downloadsDir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "Image2Midi")
            if (!downloadsDir.exists()) downloadsDir.mkdirs()

            val outputFile = File(downloadsDir, pdfFile.name)

            FileInputStream(pdfFile).use { inputStream ->
                FileOutputStream(outputFile).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }

            Log.d("PDF_DOWNLOAD", "PDF saved to: ${outputFile.absolutePath}")
            Toast.makeText(this, "PDF downloaded", Toast.LENGTH_SHORT).show()


            MediaScannerConnection.scanFile(
                applicationContext,
                arrayOf(outputFile.absolutePath),
                arrayOf("application/pdf"),
                null
            )

        } catch (e: Exception) {
            Log.e("PDF_DOWNLOAD_ERROR", "Error downloading PDF: ${e.message}")
        }
    }


    // ------ SERVER CONNECTION FUNCTIONS

    private fun selectingServer() {
        val intent = Intent(this, ServerActivity::class.java)
        selectingServerLauncher.launch(intent)
    }


    private fun uploadToPythonServer(photo: ByteArray, option: Int) {

        setPlayerBarVisibility(View.GONE)
        setPdfBarVisibility(View.GONE)

        setLoadingVisibility(View.VISIBLE)
        errorTextView?.visibility = View.GONE

        val tempFile = ImageManager.byteArrayToFile(this, photo, "temp_img.jpg")
        if (tempFile != null) {

            val requestFile = tempFile.asRequestBody("image/jpeg".toMediaTypeOrNull())
            val photoBody = MultipartBody.Part.createFormData("file", tempFile.name, requestFile)
            val optionBody = option.toString().toRequestBody("text/plain".toMediaTypeOrNull())

            lifecycleScope.launch {
                try {
                    val response = apiService.uploadImage(photoBody, optionBody)
                    if (response.isSuccessful) {

                        setLoadingVisibility(View.GONE)

                        val responseBody = response.body()

                        author = responseBody?.get("author")?.replace(":", "") ?: "???"
                        title = responseBody?.get("title")?.replace("\n", "")?.replace(":", "") ?: "???"

                        Log.d("author", author.replace(" ", "_"))

                        // midi
                        if (option == 0) {
                            responseBody?.let {

                                val startTime = it["start"] ?: ""
                                val endTime = it["end"] ?: ""

                                Log.d("test", "$startTime : $endTime")

                                val base64Midi = it["output"] ?: ""
                                val midiBytes = Base64.decode(base64Midi, Base64.DEFAULT)

                                // save midifile
                                val midiFile = saveMidiFile(midiBytes)
                                if (midiFile != null) {
                                    selectedMidiUri = Uri.fromFile(midiFile)
                                    playMidi(selectedMidiUri!!) // start playback
                                } else {
                                    Log.e("MIDI_ERROR", "Failed to save MIDI file")
                                    setErrorStr(getString(R.string.error_image))
                                }

                            }
                        }
                        // pdf
                        else if (option == 1) {
                            val base64Pdf = responseBody?.get("output") ?: ""
                            val pdfBytes = Base64.decode(base64Pdf, Base64.DEFAULT)
                            val pdfFile = savePdfFile(pdfBytes)

                            if (pdfFile != null) {
                                selectedPdfUri = Uri.fromFile(pdfFile)
                                setPdfBarVisibility(View.VISIBLE)
                            } else {
                                Log.e("PDF ERROR", "Failed to save the PDF file.")
                                setErrorStr(getString(R.string.error_pdf))
                            }
                        }

                    } else {
                        setLoadingVisibility(View.GONE)
                        Log.e(
                            "UPLOAD_ERROR",
                            "Failed to upload image: ${response.errorBody()?.string()}"
                        )
                        setErrorStr(getString(R.string.error_image))
                    }
                } catch (e: Exception) {
                    setLoadingVisibility(View.GONE)
                    Log.e("UPLOAD_EXCEPTION", "Exception while uploading image: ${e.message}")
                    setErrorStr(getString(R.string.error_server))
                }
            }
        } else {
            setLoadingVisibility(View.GONE)
            Log.e("ERROR-UPLOAD", "unable to upload the image to server")
            setErrorStr(getString(R.string.error_server))
        }
    }

    // ------ VISIBILITY SETTINGS

    private fun setHomeVisibility(visibility: Int) {

        findViewById<TextView>(R.id.title).visibility    = visibility
        findViewById<TextView>(R.id.credits).visibility  = visibility

    }


    private fun setPhotoBtnVisibility(visibility: Int) {

        findViewById<ImageButton>(R.id.home_button).visibility = visibility
        findViewById<ImageButton>(R.id.crop_button).visibility = visibility
        findViewById<LinearLayout>(R.id.server_interaction_buttons).visibility = visibility

    }


    private fun setLoadingVisibility(visibility: Int) {

        loadingSpinner.visibility = visibility
        loadingOverlay.visibility = visibility

    }


    private fun setPlayerBarVisibility(visibility: Int) {

        findViewById<LinearLayout>(R.id.player_bar).visibility = visibility
        if (visibility == View.GONE) playerManager.stop()
        else if (visibility == View.VISIBLE)
            findViewById<TextView>(R.id.auth_title).text = getString(R.string.author_title, author, title)

    }


    private fun setPdfBarVisibility(visibility: Int) {

        findViewById<LinearLayout>(R.id.pdf_bar).visibility = visibility

        if (visibility == View.VISIBLE)
            findViewById<TextView>(R.id.pdf_name).text = getString(R.string.author_title_pdf, author, title)

    }


    // ------ LAUNCHER FUNCTIONS

    private val cameraLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->

        if (result.resultCode == RESULT_OK) {

            photoUri?.let { uri ->
                val photo: Bitmap? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    val source = ImageDecoder.createSource(contentResolver, uri)
                    ImageDecoder.decodeBitmap(source)
                } else {
                    contentResolver.openInputStream(uri)?.use { inputStream ->
                        BitmapFactory.decodeStream(inputStream)
                    }
                }

                if (photo != null) {
                    val imgData = ImageManager.convertBitmapToByteArray(photo)
                    setImageView(imgData)
                }
            }

        }

    }


    private val galleryLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) {
            uri: Uri? -> uri?.let { selectedImageUri ->

        try {

            val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }

            contentResolver.openInputStream(selectedImageUri)?.use {
                    inputStream ->

                BitmapFactory.decodeStream(inputStream, null, options)
            }

            options.inSampleSize = calculateInSampleSize(options, 1024, 1024)
            options.inJustDecodeBounds = false

            val bitmap = contentResolver.openInputStream(selectedImageUri)?.use { inputStream ->
                BitmapFactory.decodeStream(inputStream, null, options)
            }

            bitmap?.let {
                val imgData = ImageManager.convertBitmapToByteArray(it)
                setImageView(imgData)
            }

        } catch (e: Exception) { setErrorStr(getString(R.string.error_gallery)) }
    }
    }


    private val croppingLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK) {
            result.data?.getStringExtra("cropped_image_uri")?.let { croppedImageUriStr ->
                val croppedUri = Uri.parse(croppedImageUriStr)

                // load the cropped image to the imageview
                contentResolver.openInputStream(croppedUri)?.use { inputStream ->
                    val croppedBitmap = BitmapFactory.decodeStream(inputStream)
                    setImageView(ImageManager.convertBitmapToByteArray(croppedBitmap))
                }
            }
        } else { setErrorStr(getString(R.string.error_cropping)) }
    }


    private val selectingServerLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK) {

            ip = result.data?.getStringExtra("IP_ADDRESS").toString()
            port = result.data?.getStringExtra("PORT").toString()

            Log.d("test", "$ip:$port")

            apiService = RetrofitClient.createRetroFit(this, ip, port)
        } else { Log.e("ERROR", "error selecting server") }
    }






}
