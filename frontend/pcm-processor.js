/**
 * TheONE AI - PCM 16kHz AudioWorklet Processor
 *
 * Runs on the audio rendering thread to downsample microphone audio
 * from the browser's native sample rate (e.g., 44.1kHz or 48kHz)
 * to 16,000 Hz signed 16-bit little-endian PCM.
 *
 * Emits binary frames of 512 samples (1024 bytes) at ~32ms intervals
 * for real-time streaming to Gnani Prisma STT v3.
 *
 * Supports pause/resume commands to prevent acoustic echo / audio leakage
 * into the STT stream while AI is thinking or speaking.
 */

class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    // sampleRate is a global in AudioWorkletGlobalScope
    this.sourceSampleRate = sampleRate || 48000;
    this.ratio = this.sourceSampleRate / this.targetSampleRate;
    this.resampleIndex = 0.0;

    this.frameSamples = 512; // 1024 bytes of 16-bit PCM
    this.outputBuffer = new Int16Array(this.frameSamples);
    this.outputIndex = 0;
    this.isPaused = false;

    this.port.onmessage = (event) => {
      const data = event.data || {};
      if (data.command === 'pause') {
        this.isPaused = true;
        this.outputIndex = 0;
        this.resampleIndex = 0.0;
      } else if (data.command === 'resume') {
        this.isPaused = false;
        this.outputIndex = 0;
        this.resampleIndex = 0.0;
      } else if (data.command === 'clear') {
        this.outputIndex = 0;
        this.resampleIndex = 0.0;
      }
    };
  }

  process(inputs, outputs, parameters) {
    // Drop all audio frames if paused during AI thinking, speaking, or settling
    if (this.isPaused) {
      return true;
    }

    const input = inputs[0];
    if (!input || input.length === 0) return true;

    // Use channel 0 (mono)
    const channelData = input[0];
    const inputLen = channelData.length;

    while (this.resampleIndex < inputLen) {
      // Linear interpolation between adjacent input samples
      const idx = Math.floor(this.resampleIndex);
      const frac = this.resampleIndex - idx;
      const s0 = channelData[idx];
      const s1 = idx + 1 < inputLen ? channelData[idx + 1] : s0;
      const sample = s0 + frac * (s1 - s0);

      // Clamp to [-1.0, 1.0] and convert to signed 16-bit PCM integer
      const clamped = Math.max(-1.0, Math.min(1.0, sample));
      const pcm16 = clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);

      this.outputBuffer[this.outputIndex++] = pcm16;

      // When 512 samples (1024 bytes) accumulated, transfer to main thread
      if (this.outputIndex >= this.frameSamples) {
        this.port.postMessage(this.outputBuffer.buffer, [this.outputBuffer.buffer]);
        this.outputBuffer = new Int16Array(this.frameSamples);
        this.outputIndex = 0;
      }

      this.resampleIndex += this.ratio;
    }

    this.resampleIndex -= inputLen;
    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);
