package com.example.remotebootlegscore

import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Response
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface ApiService {

    @Multipart
    @POST("process-image")
    suspend fun uploadImage(@Part file: MultipartBody.Part,
                            @Part("option") option: RequestBody
    ): Response<Map<String, String>>

}