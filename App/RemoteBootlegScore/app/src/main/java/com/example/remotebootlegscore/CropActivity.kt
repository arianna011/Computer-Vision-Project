package com.example.remotebootlegscore

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PointF
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import kotlin.math.pow
import kotlin.math.sqrt

class CroppingActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var bitmap: Bitmap
    private val selectedPoints = mutableListOf<PointF>()

    private lateinit var applyCropBtn: Button

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_crop)

        if (supportActionBar != null) {
            supportActionBar?.hide();
        }

        imageView = findViewById(R.id.image_to_crop)
        imageView.scaleType = ImageView.ScaleType.FIT_CENTER

        applyCropBtn = findViewById(R.id.applyCropButton)

        val imageUri = intent.getStringExtra("image_uri")?.let { Uri.parse(it) }
        if (imageUri != null) {
            // Decode the bitmap from the URI and assign it to the global 'bitmap' property
            bitmap = decodeBitmapFromUri(imageUri)  // Use the 'bitmap' property here
            Log.d("AAA", bitmap.toString())
            imageView.setImageBitmap(bitmap)
        }

        // Set up touch listener for point selection
        imageView.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_UP) {
                handleTouch(event.x, event.y)
            }
            true
        }

        applyCropBtn.setOnClickListener {
            sendCroppedBitmapBack()
        }

    }


    private fun decodeBitmapFromUri(uri: Uri): Bitmap {
        val inputStream: InputStream? = contentResolver.openInputStream(uri)
        return BitmapFactory.decodeStream(inputStream)
    }


    private fun handleTouch(x: Float, y: Float) {
        if (selectedPoints.size < 4) {
            // Add the selected point
            selectedPoints.add(PointF(x, y))
            drawPoints() // Visual feedback

            if (selectedPoints.size == 4) {
                // Once 4 points are selected, perform cropping
                performCropAndStraighten()
            }
        } else {
            Toast.makeText(this, "4 points already selected!", Toast.LENGTH_SHORT).show()
        }
    }

    private fun drawPoints() {
        // Create a mutable bitmap to draw points on
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint().apply {
            color = Color.RED
            style = Paint.Style.FILL
            strokeWidth = 10f
        }

        // Draw each selected point
        selectedPoints.forEach { point ->
            canvas.drawCircle(point.x, point.y, 10f, paint)
        }

        imageView.setImageBitmap(mutableBitmap) // Update the ImageView with the new bitmap
    }

    private fun performCropAndStraighten() {
        // Map the selected points to the original bitmap's coordinates
        val srcPoints = selectedPoints.map { point ->
            mapPointToBitmapCoordinates(point)
        }.flatMap { listOf(it.x, it.y) }.toFloatArray()

        // Use the original bitmap dimensions
        val targetWidth = bitmap.width
        val targetHeight = bitmap.height

        // Define the destination points to keep the original size
        val dstPoints = floatArrayOf(
            0f, 0f, // Top-left
            targetWidth.toFloat(), 0f, // Top-right
            targetWidth.toFloat(), targetHeight.toFloat(), // Bottom-right
            0f, targetHeight.toFloat() // Bottom-left
        )

        // Apply the transformation matrix
        val transformationMatrix = Matrix()
        transformationMatrix.setPolyToPoly(srcPoints, 0, dstPoints, 0, 4)

        // Create a new bitmap with the original dimensions
        val transformedBitmap = Bitmap.createBitmap(
            targetWidth,
            targetHeight,
            bitmap.config ?: Bitmap.Config.ARGB_8888
        )

        // Apply transformation to the full image
        val canvas = Canvas(transformedBitmap)
        canvas.drawBitmap(bitmap, transformationMatrix, Paint(Paint.ANTI_ALIAS_FLAG))

        // Display the result (or save it)
        imageView.setImageBitmap(transformedBitmap)
        bitmap = transformedBitmap

        Toast.makeText(this, "Crop and straighten completed!", Toast.LENGTH_SHORT).show()
    }

    private fun distance(p1: PointF, p2: PointF): Float {
        return sqrt(((p2.x - p1.x).toDouble().pow(2.0) + (p2.y - p1.y).toDouble().pow(2.0))).toFloat()
    }

    private fun mapPointToBitmapCoordinates(point: PointF): PointF {
        // Map the point from ImageView coordinates to Bitmap coordinates
        val imageMatrix = imageView.imageMatrix
        val values = FloatArray(9)
        imageMatrix.getValues(values)

        val scaleX = values[Matrix.MSCALE_X]
        val scaleY = values[Matrix.MSCALE_Y]
        val transX = values[Matrix.MTRANS_X]
        val transY = values[Matrix.MTRANS_Y]

        val bitmapX = (point.x - transX) / scaleX
        val bitmapY = (point.y - transY) / scaleY

        return PointF(bitmapX, bitmapY)
    }

    @SuppressLint("UnsafeIntentLaunch")
    private fun sendCroppedBitmapBack() {
        val bitmapUri = saveBitmapToFile(bitmap)
        // Return result via intent
        val resultIntent = intent
        resultIntent.putExtra("cropped_image_uri", bitmapUri.toString())
        setResult(RESULT_OK, resultIntent)
        finish()
    }

    override fun onDestroy() {
        super.onDestroy()
        bitmap.recycle() // Avoid memory leaks
    }

    private fun saveBitmapToFile(bitmap: Bitmap): Uri {
        // Create a temporary file
        val file = File(cacheDir, "temp_image.jpg")
        FileOutputStream(file).use { outStream ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outStream)
        }
        return FileProvider.getUriForFile(this, "com.example.remotebootlegscore.fileprovider", file)
    }
}