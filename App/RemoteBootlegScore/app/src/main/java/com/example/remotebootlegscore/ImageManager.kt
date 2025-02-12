package com.example.remotebootlegscore

import android.content.Context
import android.graphics.Bitmap
import android.icu.text.SimpleDateFormat
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.Date
import java.util.Locale


object ImageManager {

    fun createImageFile(context: Context): File? {
        return try {

            val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(
                Date()
            )
            val storageDir: File? = context.getExternalFilesDir(null)
            File.createTempFile("JPEG_${timeStamp}_", ".jpg", storageDir)

        } catch (e: Exception) {

            Log.e("FILE_CREATION", "Failed to create image file", e)
            null

        }
    }


    fun convertBitmapToByteArray(bitmap: Bitmap): ByteArray {

        val byteArrayOutputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream)
        return byteArrayOutputStream.toByteArray()

    }


    fun byteArrayToFile(context: Context, byteArray: ByteArray, fileName: String): File? {

        return try {

            val file = File(context.cacheDir, fileName)
            file.outputStream().use { outputStream ->
                outputStream.write(byteArray)
            }
            file

        } catch (e: Exception) {

            Log.e("FILE_ERROR", "Error converting byte array to file: ${e.message}")
            null

        }

    }
}