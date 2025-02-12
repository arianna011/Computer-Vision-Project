package com.example.remotebootlegscore

import android.content.Context
import android.media.MediaPlayer
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.util.Log

class PlayerManager(private val context: Context,
                    private val onStateChange: (isPlaying: Boolean) -> Unit,
                    private val onProgressUpdate: (progress: Int, duration: Int) -> Unit) {

    private var mediaPlayer: MediaPlayer? = null
    private var isPaused: Boolean = false
    private var handler = Handler(Looper.getMainLooper())
    private var currentUri: Uri? = null


    fun play(uri: Uri, forceChange: Boolean = false) {

        if (forceChange) stop()

        if (mediaPlayer == null || currentUri != uri) {

            Log.d("ue", "im in!")

            currentUri = uri
            mediaPlayer = MediaPlayer().apply {
                setDataSource(context, uri)
                prepare()
                start()
                onStateChange(true)
                startProgressUpdater()

                setOnCompletionListener {
                    isPaused = true
                    onStateChange(false)
                    seekTo(0)
                }
            }
            onStateChange(true)

        } else if (isPaused) {

            mediaPlayer?.start()
            isPaused = false
            onStateChange(true)
            startProgressUpdater()

        }

    }


    fun pause() {

        mediaPlayer?.let {
            if (it.isPlaying) {
                it.pause()
                isPaused = true
                onStateChange(false)
            }
        }

    }


    fun stop() {

        mediaPlayer?.release()
        mediaPlayer = null
        currentUri = null
        isPaused = false
        onStateChange(false)

    }


    fun isPlaying(): Boolean { return mediaPlayer?.isPlaying == true }


    fun seekTo(position: Int) { mediaPlayer?.seekTo(position) }


    private fun startProgressUpdater() {

        handler.postDelayed(object : Runnable {

            override fun run() {
                mediaPlayer?.let {
                    if (it.isPlaying) {

                        onProgressUpdate(it.currentPosition, it.duration)
                        handler.postDelayed(this, 500)

                    }
                }
            }

        }, 500)

    }

}