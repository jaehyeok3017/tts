package com.example.tts

import android.Manifest
import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Process
import android.util.Log
import androidx.annotation.WorkerThread
import com.example.tts.databinding.ActivityMainBinding
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor
import java.nio.FloatBuffer
import kotlin.concurrent.fixedRateTimer

class MainActivity : AppCompatActivity() {
    private var mModule : Module? = null
    private lateinit var viewBinding : ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        viewBinding.recBtn.setOnClickListener {
            viewBinding.recBtn.isEnabled = false
            var time = AUDIO_LEN_IN_SECOND
            fixedRateTimer(initialDelay = 0, period = 1000){
                if (time == 0) {
                    this.cancel()
                    runOnUiThread { viewBinding.textView.text="Start" }
                }

                else {
                    runOnUiThread { viewBinding.recBtn.text = "${time} sec" }
                    time -= 1
                }
            }
            Thread{
                record()
            }.start()
        }

        // 권한을 받아와줌 --> Record_audio (음성녹음) [minSdk 24 --> 21은 적어서 X]
        requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 0xff)
    }

    @SuppressLint("MissingPermission")
    @WorkerThread
    fun record(){
        Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO) // 오디오라는 것을 명시해줌
        val bufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO, // mono는 단일 스피커, 2개 이상 : 스테레오
            AudioFormat.ENCODING_PCM_16BIT // 한 샘플을 16bit 로 부호화 시킬 것임
        )
        val record = AudioRecord(
            MediaRecorder.AudioSource.DEFAULT, // 일반 마이크
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO, // mono는 단일 스피커, 2개 이상 : 스테레오
            AudioFormat.ENCODING_PCM_16BIT, // 한 샘플을 16bit 로 부호화 시킬 것임
            bufferSize
        )

        if(record.state != AudioRecord.STATE_INITIALIZED){ // audio X
            Log.e(LOG_TAG, "Audio Record cna't initialized")
            return
        }

        record.startRecording()
        var readed : Long = 0
        var recordOff = 0
        val audioBuffer = ShortArray(bufferSize / 2) //16bit이므로 short 상관 x - 전체버퍼의 절반
        val recordBuffer = ShortArray(RECORDING_LENGTH) // 전체 버퍼

        // 일정 메모리만 받아오고, 버리고 그런 방식으로 적용 --> 속도 지연 방지
        while(readed < RECORDING_LENGTH){
            val curreaded = record.read(audioBuffer, 0, audioBuffer.size) // audio buffer에 받아줌
            readed += curreaded.toLong()
            System.arraycopy(audioBuffer, 0, recordBuffer, recordOff, curreaded) // recordbuffer에 써줌
            recordOff += curreaded
        }

        // 전처리 과정
        record.stop() // while문 빠져나왔을 시 ( 일정 길이 up )
        record.release() // audio 상태를 계속 점유하는 것을 방지 : 마이크 권한

        val floatInputBuffer = FloatArray(RECORDING_LENGTH)

        // pytorch에서 오디오를 처리할 때 다양한 해상도에서 -1 ~ 1까지 표현되는 값 바꿔버림
        for(i in 0 until RECORDING_LENGTH){
            floatInputBuffer[i] = recordBuffer[i] / Short.MAX_VALUE.toFloat()
        }

        val result = recognize(floatInputBuffer)
        runOnUiThread { // 쓰레드 안에서 돌아가므로
            viewBinding.recBtn.isEnabled = true
            viewBinding.textView.text = result
        }
    }

    fun recognize(floatInputBuffer: FloatArray): String{
        if(mModule == null) {
            mModule = LiteModuleLoader.load(
                Utils.assetFilePath(applicationContext, "wav2vec2.ptl")
            )
        }

            val wav2vecInput = FloatArray(RECORDING_LENGTH) // 해상도를 높여 더블로 작성해줌
            for(n in 0 until RECORDING_LENGTH)
                wav2vecInput[n] = floatInputBuffer[n]
            val inTensorBuffer = Tensor.allocateFloatBuffer(RECORDING_LENGTH) // 변수는 메모리할당을 한공간에 몰아주는게 좋을 수 있음
            for(ipt in wav2vecInput) inTensorBuffer.put(ipt)
            val inTensor = Tensor.fromBlob(
                inTensorBuffer,
                longArrayOf(1, RECORDING_LENGTH.toLong()) // 1차원, 길이
            )
            return mModule!!.forward(IValue.from(inTensor)).toStr()

    }

    companion object{
        private val LOG_TAG = MainActivity::class.java.simpleName // 어느 페이지에서 에러가 났는지
        private const val SAMPLE_RATE = 16000 // 1초에 몇개의 점을 셀 것인가?
        private const val AUDIO_LEN_IN_SECOND = 6 // Recoding 길이
        private const val RECORDING_LENGTH = SAMPLE_RATE * AUDIO_LEN_IN_SECOND // 프로그래밍 과정에서 얼마나 가져올건가?
    }
}