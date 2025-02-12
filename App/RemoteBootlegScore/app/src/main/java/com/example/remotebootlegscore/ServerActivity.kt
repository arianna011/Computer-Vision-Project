package com.example.remotebootlegscore

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

class ServerActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_server)

        findViewById<Button>(R.id.send_button).setOnClickListener {
            val ip = findViewById<EditText>(R.id.ip_address).text.toString().trim()
            val port = findViewById<EditText>(R.id.port).text.toString().trim()

            if (validateInput(ip, port)) {
                Toast.makeText(this, "New IP and PORT selected", Toast.LENGTH_SHORT).show()
                val resultIntent = Intent()
                resultIntent.putExtra("IP_ADDRESS", ip)
                resultIntent.putExtra("PORT", port)
                setResult(RESULT_OK, resultIntent)
                finish()
            }
        }
    }


    private fun validateInput(ip: String, port: String): Boolean{
        if (ip.isEmpty() || port.isEmpty()) {
            Toast.makeText(this, "Please enter both IP and Port", Toast.LENGTH_SHORT).show()
            return false
        }

        // Validate IP (simple check for dots)
        if (!android.util.Patterns.IP_ADDRESS.matcher(ip).matches()) {
            Toast.makeText(this, "Invalid IP Address", Toast.LENGTH_SHORT).show()
            return false
        }

        // Validate Port (must be a number between 1-65535)
        val portNumber = port.toIntOrNull()
        if (portNumber == null || portNumber !in 1..65535) {
            Toast.makeText(this, "Invalid Port Number", Toast.LENGTH_SHORT).show()
            return false
        }

        return true
    }
}