import React, { useState, useRef } from 'react';
import Layout from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Mic, Upload } from "lucide-react";
import { Alert } from "@/components/ui/alert";

const TranslatePage: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [translatedText, setTranslatedText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/wav' });
        const file = new File([audioBlob], 'recording.wav', { type: 'audio/wav' });
        setAudioFile(file);
        setError(null);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setError(null);
    } catch (err) {
      console.error('Error accessing microphone:', err);
      setError('Error accessing microphone. Please make sure you have granted permission.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const validateFile = (file: File): boolean => {
    // Check file type
    const validTypes = ['audio/mp3', 'audio/mpeg', 'audio/wav', 'audio/wave'];
    if (!validTypes.includes(file.type)) {
      setError('Please upload an MP3 or WAV file');
      return false;
    }

    // Check file size (max 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB in bytes
    if (file.size > maxSize) {
      setError('File size must be less than 10MB');
      return false;
    }

    return true;
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    setError(null);

    if (file) {
      if (validateFile(file)) {
        setAudioFile(file);
        setError(null);
      }
    } else {
      setError('No file selected');
    }
  };

  const translateAudio = async () => {
    if (!audioFile) {
      setError('Please upload an audio file or record audio first');
      return;
    }

    setIsLoading(true);
    setError(null);
    const form = new FormData();
    form.append("with_diarization", "false");
    form.append("num_speakers", "1");
    form.append("model", "saaras:flash");
    form.append("file", audioFile);

    try {
      const response = await fetch('https://api.sarvam.ai/speech-to-text-translate', {
        method: 'POST',
        headers: {
          'api-subscription-key': '33849073-620d-4062-87e1-b4d05e7a28a9',
        },
        body: form,
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(errorData || 'Translation failed');
      }

      const data = await response.json();
      setTranslatedText(data.transcript);
      setError(null);
    } catch (err) {
      console.error('Translation error:', err);
      setError(err instanceof Error ? err.message : 'Translation failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Layout>
      <div className="container mx-auto p-4 max-w-2xl">
        <h1 className="text-2xl font-bold mb-6">Audio Translation</h1>
        
        <Card className="p-6 mb-6">
          <div className="space-y-4">
            {error && (
              <Alert variant="destructive" className="mb-4">
                {error}
              </Alert>
            )}
            
            <div className="flex gap-4">
              <Button
                onClick={isRecording ? stopRecording : startRecording}
                variant={isRecording ? "destructive" : "default"}
                className="flex-1"
              >
                <Mic className="w-5 h-5 mr-2" />
                {isRecording ? 'Stop Recording' : 'Start Recording'}
              </Button>

              <div className="flex-1">
                <input
                  type="file"
                  accept=".mp3,.wav,audio/mp3,audio/mpeg,audio/wav"
                  onChange={handleFileUpload}
                  className="hidden"
                  ref={audioInputRef}
                />
                <Button
                  onClick={() => audioInputRef.current?.click()}
                  variant="outline"
                  className="w-full"
                >
                  <Upload className="w-5 h-5 mr-2" />
                  Upload Audio
                </Button>
              </div>
            </div>

            {audioFile && (
              <div className="text-sm text-muted-foreground">
                Selected file: {audioFile.name} ({(audioFile.size / (1024 * 1024)).toFixed(2)} MB)
              </div>
            )}

            <Button
              onClick={translateAudio}
              disabled={!audioFile || isLoading}
              className="w-full"
            >
              {isLoading ? 'Translating...' : 'Translate Audio'}
            </Button>

            {translatedText && (
              <div className="mt-4">
                <h2 className="text-lg font-semibold mb-2">Translation Result:</h2>
                <Textarea
                  value={translatedText}
                  readOnly
                  className="w-full min-h-[200px]"
                />
              </div>
            )}
          </div>
        </Card>
      </div>
    </Layout>
  );
};

export default TranslatePage; 