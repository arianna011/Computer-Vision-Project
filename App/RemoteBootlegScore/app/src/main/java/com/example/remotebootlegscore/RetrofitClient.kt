package com.example.remotebootlegscore

import android.content.Context
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.io.InputStream
import java.security.KeyStore
import java.security.cert.CertificateFactory
import java.util.concurrent.TimeUnit
import javax.net.ssl.SSLContext
import javax.net.ssl.TrustManagerFactory
import javax.net.ssl.X509TrustManager

object RetrofitClient {


    fun createRetroFit(context: Context, ip: String = "192.168.196.227", port: String = "6969"): ApiService {

        return Retrofit.Builder()
            .baseUrl("https://$ip:$port/")
            .addConverterFactory(GsonConverterFactory.create())
            .client(getTrustingOkHttpClient(context))
            .build()
            .create(ApiService::class.java)

    }


    private fun getTrustingOkHttpClient(context: Context): OkHttpClient {

        // load the self-signed certificate from raw resources
        val cf: CertificateFactory = CertificateFactory.getInstance("X.509")
        val certInputStream: InputStream = context.resources.openRawResource(R.raw.server)
        val cert = cf.generateCertificate(certInputStream)

        // add the certificate to the keystore
        val keyStore = KeyStore.getInstance(KeyStore.getDefaultType()).apply {
            load(null, null)
            setCertificateEntry("server", cert)
        }

        // create a trust manager using the keystore
        val trustManagerFactory = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm()).apply {
            init(keyStore)
        }
        val trustManagers = trustManagerFactory.trustManagers
        val x509TrustManager = trustManagers[0] as X509TrustManager

        // set up sslcontext with the certificate
        val sslContext = SSLContext.getInstance("TLS").apply {
            init(null, arrayOf(x509TrustManager), null)
        }

        // return an okhttpclient with sslcontext configured
        return OkHttpClient.Builder()
            .sslSocketFactory(sslContext.socketFactory, x509TrustManager)
            .hostnameVerifier { _, _ -> true }
            .connectTimeout(20, TimeUnit.SECONDS)
            .readTimeout(20, TimeUnit.SECONDS)
            .writeTimeout(20, TimeUnit.SECONDS)
            .build()

    }

}