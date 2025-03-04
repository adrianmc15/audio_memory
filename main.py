
# main (version 9.1):
# > start and provision the microphone
# > after 3 minutes, make a call to AssemblyAI, which will transcribe the audio
# > once the transcription is complete, use the established method to link to the original voices
# > then allow queries from the LLM


import sys
import pyaudio
import wave
import threading
import assemblyai as aai
import time
from datetime import datetime, timedelta
import json
import datetime as dt
import os
import json
from pydub import AudioSegment
import heapq
from dotenv import load_dotenv

load_dotenv(".env")
# AssemblyAI API setup
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
config = aai.TranscriptionConfig(
    speaker_labels=True,
)
# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 10  # Record for 3 minutes
RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"
TEMP_PATH = "data/temp"
LIBRARY_PATH="data/library"
OUTPUT_PATH = "data/output"
TOP_N_UTTERANCES = 6  # Number of longest utterances to use per speaker
MIN_UTTERANCE_DURATION = 1000  # Minimum duration of an utterance in milliseconds

OUTPUT_FILENAME_TEMPLATE = "{}.wav"

def diarize_file(input_path, input_containing_path=RAW_PATH, processed_folder_path=PROCESSED_PATH):
    # for file in os.listdir(raw_path):
    try:
        os.mkdir(os.path.join(processed_folder_path, input_path.split(".")[0]))
    except FileExistsError:
        print("folder exists")

    audio_file = os.path.join(input_containing_path,input_path)

    transcript = aai.Transcriber().transcribe(audio_file, config)
    
    # save the transcript as a json file, with the structure: recording { [ {word,start_time,end_time,speaker_number} , ...] }
    words = []
    for utterance in transcript.utterances:
        for word in utterance.words:
            words.append({
                "word": word.text,
                "start_time": word.start,
                "end_time": word.end,
                "speaker": utterance.speaker
            })

    # add the recording path
    data = {
        "file": audio_file,
        "words": words
    }

    # Save the JSON object to a file
    output_file = os.path.join(processed_folder_path, input_path.split(".")[0], "transcript.json")
    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)

    return output_file, transcript


def record_audio_chunk():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    os.makedirs("data/raw", exist_ok=True)

    print("Recording started...")

    # while True:
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    timestamp_dt = datetime.now()
    timestamp = timestamp_dt.strftime("%Y%m%d_%H%M%S")
    file_name = OUTPUT_FILENAME_TEMPLATE.format(timestamp)

    with wave.open(f"data/raw/{file_name}", 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved to {file_name}. Starting transcription...")

    return file_name, timestamp_dt


def extract_speaker_utterances(transcript_path,raw_path=RAW_PATH,min_utterance_duration=MIN_UTTERANCE_DURATION):
    """Extract and sort utterances by duration for each speaker."""
    print(f"\nProcessing transcript: {transcript_path}")
    
    with open(transcript_path, 'r') as f:
        data = json.load(f)
    
    original_filename = os.path.basename(data['file'])
    audio_file = os.path.join(raw_path, original_filename)
    print(f"Looking for audio file: {audio_file}")
    
    # Group words by speaker and utterance
    speaker_utterances = {}
    current_speaker = None
    current_utterance = []
    max_pause = 1000  # Maximum pause between words in the same utterance
    
    print(f"Total words in transcript: {len(data['words'])}")
    
    for i, word in enumerate(data['words']):
        if current_speaker != word['speaker'] or (
            current_utterance and 
            word['start_time'] - current_utterance[-1]['end_time'] > max_pause
        ):
            # Save previous utterance if it exists
            if current_utterance:
                duration = current_utterance[-1]['end_time'] - current_utterance[0]['start_time']
                if duration >= min_utterance_duration:
                    if current_speaker not in speaker_utterances:
                        speaker_utterances[current_speaker] = []
                    # Store as tuple of (duration, utterance) for sorting
                    heapq.heappush(speaker_utterances[current_speaker], 
                                 (-duration, current_utterance))
                    print(f"Added utterance for speaker {current_speaker} with duration {duration}ms")
            
            current_utterance = [word]
            current_speaker = word['speaker']
        else:
            current_utterance.append(word)
    
    # Add the last utterance
    if current_utterance:
        duration = current_utterance[-1]['end_time'] - current_utterance[0]['start_time']
        if duration >= min_utterance_duration:
            if current_speaker not in speaker_utterances:
                speaker_utterances[current_speaker] = []
            heapq.heappush(speaker_utterances[current_speaker], 
                          (-duration, current_utterance))
            print(f"Added final utterance for speaker {current_speaker} with duration {duration}ms")
    
    print("\nSpeaker statistics:")
    for speaker, utterances in speaker_utterances.items():
        print(f"Speaker {speaker}: {len(utterances)} utterances")
    
    return speaker_utterances, audio_file


def get_top_n_utterances_transcript(transcript_path, top_n_utterances=TOP_N_UTTERANCES, raw_path=RAW_PATH):
    """Gets the top N longest utterances from each speaker in a transcript.
       Returns:
       - A list of speakers
       - Their top N longest utterances as audio objects
       - Metadata for each utterance (original file, timestamps, speaker ID)
    """
    try:
        print(f"\nProcessing file: {transcript_path}")
        
        # Extract utterances and associated audio file
        speaker_utterances, audio_file = extract_speaker_utterances(transcript_path, raw_path=raw_path)
        
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file not found: {audio_file}")
            return None  # Skip processing if audio file is missing
        
        print(f"Loading audio file: {audio_file}")    
        audio = AudioSegment.from_wav(audio_file)
        print(f"Audio duration: {len(audio)}ms")
        
        results = []  # List to store speaker data

        for speaker_id, utterances_heap in speaker_utterances.items():
            print(f"\nProcessing speaker {speaker_id}")
            print(f"Total utterances available: {len(utterances_heap)}")
            
            # Extract top N longest utterances
            top_utterances_audio = []
            metadata = []
            
            for _ in range(min(top_n_utterances, len(utterances_heap))):
                try:
                    if utterances_heap:
                        duration, utterance = heapq.heappop(utterances_heap)
                        duration = -duration  # Convert back to positive
                        print(f"Selected utterance with duration: {duration}ms")
                        
                        # Extract audio segment
                        start_ms = int(utterance[0]['start_time'])
                        end_ms = int(utterance[-1]['end_time'])
                        utterance_audio = audio[start_ms:end_ms]  # Extract audio clip
                        
                        top_utterances_audio.append(utterance_audio)
                        metadata.append({
                            "speaker_id": speaker_id,
                            "transcript_path": transcript_path,
                            "audio_path": os.path.basename(audio_file),
                            "start_time": start_ms,
                            "end_time": end_ms,
                            "duration_ms": duration
                        })
                except Exception as e:
                    print(f"Error processing utterance: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    continue
            
            results.append({
                "speaker_id": speaker_id,
                "utterances_audio": top_utterances_audio,  # Now contains AudioSegment objects
                "metadata": metadata
            })

        return results, speaker_utterances  # Return structured data with extracted audio clips

    except Exception as e:
        print(f"Error processing {transcript_path}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None


def create_combined_audio(utterance_data, top_n_utterances=TOP_N_UTTERANCES, raw_path=RAW_PATH, library_path=LIBRARY_PATH):
    """Create a combined audio file using the longest utterances from both the speaker library and new transcripts.

    Args:
        utterance_data (object)
        top_n_utterances (int): Number of top utterances to use per speaker.
        raw_path (str): Path to raw audio files.
        library_path (str): Path to the speaker library.

    Returns:
        combined (AudioSegment): The final combined audio.
        speaker_mapping (list): List of dictionaries mapping speakers to their audio segments.
    """
    speaker_mapping = []
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=1000)  # 1 second of silence between utterances

    speak_end_time = 0
    prev_speak_end_time = 0

    # Process speaker library
    for speaker_folder in os.listdir(library_path):
        speaker_dir = os.path.join(library_path, speaker_folder)

        if not os.path.isdir(speaker_dir):
            continue

        speaker_id = speaker_folder.replace("speaker_", "")
        metadata_path = os.path.join(speaker_dir, "metadata.json")

        # utterances_path = os.path.join(speaker_dir, "utterances.wav")

        if not os.path.exists(metadata_path):
            print(f"Skipping {speaker_folder}: Missing metadata or audio.")
            continue

        print(f"\nProcessing speaker {speaker_id} from library")

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Load audio
        # speaker_audio = AudioSegment.from_wav(utterances_path)

        # Sort utterances by duration (descending)
        top_utterances = metadata[:top_n_utterances]

        for i, utterance in enumerate(top_utterances):
            start_ms = int(utterance["start_time"])
            end_ms = int(utterance["end_time"])
            
            print(f"Adding library utterance {i+1}/{len(top_utterances)} "
                  f"(duration: {end_ms - start_ms}ms)")

            utterance_audio_path = os.path.join(speaker_dir,utterance["utterance_file"])
            sample = AudioSegment.from_wav(utterance_audio_path)
            combined += sample + silence

            speak_end_time += (end_ms - start_ms) + 1000

        speaker_mapping.append({
            "transcript_path": utterance["transcript_path"],
            "speaker_id": speaker_id,
            "start_time": prev_speak_end_time,
            "end_time": speak_end_time,
            "source": "library"
        })

        prev_speak_end_time = speak_end_time


    # put together the new audio file section
    for speaker in utterance_data:
        speaker_id = speaker["speaker_id"]
        audio_files = speaker["utterances_audio"]
        total_metadata = speaker["metadata"]

        for i, utterance_audio in enumerate(audio_files):
            metadata_i = total_metadata[i]
            start_ms = metadata_i["start_time"]
            end_ms = metadata_i["end_time"]

            combined += utterance_audio + silence

            speak_end_time += (end_ms - start_ms) + 1000

        speak_start_time = prev_speak_end_time

        speaker_mapping.append({
            "transcript_path": metadata_i["transcript_path"],
            "audio_path": metadata_i["audio_path"],
            "speaker_id": speaker_id,
            "start_time": speak_start_time,
            "end_time": speak_end_time,
            "source": "input"
        })

        prev_speak_end_time = speak_end_time

    print(f"\nFinal combined audio duration: {len(combined)}ms")
    print(f"Total speaker mappings created: {len(speaker_mapping)}")
    
    if not speaker_mapping:
        raise ValueError("No valid audio samples could be processed")
    
    return combined, speaker_mapping


def diarize_combined_audio(combined_audio_path,output_path=OUTPUT_PATH):
    """Diarize the combined audio file using AssemblyAI."""
    print(f"Diarizing combined audio file: {combined_audio_path}")
    transcript = aai.Transcriber().transcribe(combined_audio_path, config)
    
    # Save the diarized transcript to a JSON file
    diarized_transcript = {
        "file": combined_audio_path,
        "utterances": [
            {
                "start_time": utterance.start,
                "end_time": utterance.end,
                "speaker": utterance.speaker,
                "text": utterance.text
            }
            for utterance in transcript.utterances
        ]
    }
    
    output_file = os.path.join(output_path, "diarized_combined_audio.json")
    with open(output_file, 'w') as json_file:
        json.dump(diarized_transcript, json_file, indent=4)
    
    print(f"Diarized transcript saved to: {output_file}")
    return diarized_transcript



def update_transcripts(speaker_map):

    # IMPORTANT!!!! THIS UPDATE NEEDS TO BE MADE
    # TODO: update the unified_label to prioritize the existing
    # library name

    for speaker in speaker_map:
        # open the transcript_file
        file = speaker["transcript_path"]
        with open(file, 'r') as f:
            data = json.load(f)

        for word in data["words"]:
            if word["speaker"] == speaker["speaker_id"]:
                word["final_speaker"] = speaker["final_speaker"]

        # save the file
        with open(file, 'w') as f:
            json.dump(data, f, indent=4)

def save_utterances_to_library(speaker_data, library_path=LIBRARY_PATH):
    """Saves extracted speaker utterances as audio files in separate speaker folders, 
    each containing a single metadata.json file listing its utterances.

    Args:
        speaker_data (list): List of dictionaries containing speaker ID, audio utterances, and metadata.
        library_path (str): Path to save extracted utterances.
    """
    if not os.path.exists(library_path):
        os.makedirs(library_path)

    for speaker in speaker_data:
        speaker_id = speaker["speaker_id"]
        utterances_audio = speaker["utterances_audio"]
        metadata = speaker["metadata"]

        try:
            # Create a folder for the speaker
            speaker_folder = os.path.join(library_path, f"{speaker_id}")
            os.makedirs(speaker_folder, exist_ok=False)
        except Exception as e:
            print(f"Speaker {speaker_id} path already exists")
            continue

        speaker_metadata = []  # Store metadata for this speaker

        for i, (audio_segment, meta) in enumerate(zip(utterances_audio, metadata)):
            start_time = meta["start_time"]
            end_time = meta["end_time"]

            # Define filename
            utterance_filename = f"utterance_{i+1}.wav"
            utterance_path = os.path.join(speaker_folder, utterance_filename)

            # Save audio segment
            audio_segment.export(utterance_path, format="wav")
            print(f"Saved: {utterance_path}")

            # Append metadata entry for this speaker
            speaker_metadata.append({
                "start_time": start_time,
                "end_time": end_time,
                "utterance_file": utterance_filename,
                "transcript_path": meta.get("transcript_path"),
                "audio_path": meta.get("audio_path"),
            })

        # Save speaker-specific metadata.json
        metadata_path = os.path.join(speaker_folder, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(speaker_metadata, f, indent=4)

        print(f"Saved metadata file for Speaker {speaker_id}: {metadata_path}")


def update_speaker_library(updated_speaker_mapping, speaker_data):
    """Find which speakers are not already present in the speaker library
        and create new library entries for the new speakers
    """
    # requirements:
    # a. the names of the existing library folders will be kept the same
    # b. add any new speakers to the library, with new labels according to the existing ones
    
    # update speaker_data to only include "new" speakers

    # get num total speakers in speaker_data (this is the combined list of total speakers assessed)
    
    # for each updated_speaker, find the match in speaker_data based on the original speaker and the transcript location
        # add this to a new_speaker_data object 

    new_speaker_data = []
    new_speakers_list = []

    print(speaker_data)

    for segment in updated_speaker_mapping:
        if segment["new_speaker"] and segment["final_speaker"] not in new_speakers_list:
            new_speakers_list.append(segment["final_speaker"])
            
            # find the match in speaker_data
            for speaker_data_item in speaker_data:
                if speaker_data_item["metadata"][0]["transcript_path"] == segment["transcript_path"] and speaker_data_item["speaker_id"] == segment["speaker_id"]:
                    speaker_data_item["speaker_id"] = segment["final_speaker"]
                    new_speaker_data.append(speaker_data_item)
                    continue

    # call save_utterances_to_library
    save_utterances_to_library(new_speaker_data)


def link_unified_mappings(combined_diarized_transcript, speaker_mapping):
    # Iterate through each segment in the speaker_mapping
    for segment in speaker_mapping:
        segment_start = segment["start_time"]
        segment_end = segment["end_time"]
        
        # Initialize a dictionary to count the speaking time for each speaker in this segment
        speaker_time_counts = {}
        
        # Iterate through each utterance in the combined_diarized_transcript
        for utterance in combined_diarized_transcript["utterances"]:
            utterance_start = utterance["start_time"]
            utterance_end = utterance["end_time"]
            speaker = utterance["speaker"]
            
            # Calculate the overlap between the utterance and the segment
            overlap_start = max(segment_start, utterance_start)
            overlap_end = min(segment_end, utterance_end)
            
            # If there is an overlap, calculate the duration and add it to the speaker's total
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                if speaker in speaker_time_counts:
                    speaker_time_counts[speaker] += overlap_duration
                else:
                    speaker_time_counts[speaker] = overlap_duration
        
        # Determine the speaker with the maximum speaking time in this segment
        if speaker_time_counts:
            unified_speaker = max(speaker_time_counts, key=speaker_time_counts.get)
        else:
            unified_speaker = None  # No speaker found in this segment
        
        # Add the unified_speaker to the segment
        segment["unified_speaker"] = unified_speaker

    library_speakers_unified = {}

    for utterance in speaker_mapping:
        if utterance["source"] == "library":
            if utterance["unified_speaker"] not in library_speakers_unified:
                library_speakers_unified[utterance["unified_speaker"]] = utterance["speaker_id"]

    new_speakers_unified = []

    # Assign final_speaker labels
    new_speakers_unified = []
    for utterance in speaker_mapping:
        if utterance["unified_speaker"] in library_speakers_unified.keys():
            # If the unified_speaker is from the library, use the original speaker_id
            utterance["final_speaker"] = library_speakers_unified[utterance["unified_speaker"]]
            utterance["new_speaker"] = False
        else:
            if utterance["unified_speaker"] not in new_speakers_unified:
                # Assign the next sequential alphabet letter
                utterance["final_speaker"] = chr(ord('A') + len(library_speakers_unified) + len(new_speakers_unified))
                new_speakers_unified.append(utterance["unified_speaker"])
                utterance["new_speaker"] = True
            else:
                # If the unified_speaker is already in new_speakers_unified, reuse the assigned letter
                utterance["final_speaker"] = chr(ord('A') + len(library_speakers_unified) + new_speakers_unified.index(utterance["unified_speaker"]))
                utterance["new_speaker"] = True
    
    return speaker_mapping


import json
from datetime import datetime, timedelta

def append_to_transcript(json_file_location, start_time, transcript_file_location):
    """
    Processes a JSON file containing word timings and appends the formatted transcript to a file.

    Args:
        json_file_location (str): Path to the JSON file.
        start_time (datetime): A datetime object representing the start time of the recording.
        transcript_file_location (str): Path to the transcript file where the output will be appended.
    """
    # Load the JSON data
    with open(json_file_location, 'r') as file:
        data = json.load(file)

    # Ensure words are sorted by start_time
    data['words'].sort(key=lambda w: w['start_time'])

    # Initialize variables
    current_speaker = None
    current_sentence = []
    sentences = []
    first_word = None  # Track the first word of each sentence for timestamping

    # Process each word
    for word_info in data['words']:
        word = word_info['word']
        speaker = word_info['final_speaker']

        # If the speaker changes, finalize the current sentence
        if speaker != current_speaker:
            if current_sentence:
                sentences.append((current_speaker, ' '.join(current_sentence), first_word))
                current_sentence = []
            current_speaker = speaker
            first_word = word_info  # Track first word for timestamp

        # Add the word to the current sentence
        current_sentence.append(word)

    # Append the last sentence if it exists
    if current_sentence:
        sentences.append((current_speaker, ' '.join(current_sentence), first_word))

    # Open the transcript file in append mode
    with open(transcript_file_location, 'a') as transcript_file:
        # Format and append each sentence to the transcript file
        for speaker, sentence, first_word_info in sentences:
            # Use the start_time of the first word in the sentence
            timestamp = start_time + timedelta(milliseconds=first_word_info['start_time'])
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

            # Extract the speaker's name properly
            transcript_speaker = speaker.split("_")[1] if "_" in speaker else speaker

            # Format the line
            line = f"[{timestamp_str}] - {transcript_speaker}: {sentence}\n"

            # Append the line to the transcript file
            transcript_file.write(line)



def process_recording(word_diarized_path, timestamp_dt, raw_path=RAW_PATH, processed_path=PROCESSED_PATH,temp_path=TEMP_PATH, output_path=OUTPUT_PATH,top_n_utterances=TOP_N_UTTERANCES):
    transcript_file_location = "data/output/transcript.txt"

    # create a file combined with the speaker library and the new files from the 5-minute file
    utterance_data, utterance_transcript = get_top_n_utterances_transcript(word_diarized_path, top_n_utterances=top_n_utterances)

    # create a file combined with the speaker library and the new files from the 5-minute file
    combined_audio, speaker_mapping = create_combined_audio(utterance_data=utterance_data,top_n_utterances=top_n_utterances)

    # save the combined_data:
    os.makedirs(processed_path, exist_ok=True)
    combined_path = os.path.join(output_path, "combined_samples.wav")
    combined_audio.export(combined_path, format="wav")
    print(f"Combined audio saved to: {combined_path}")

    # update the transcripts
    diarized_combined = diarize_combined_audio(combined_path,output_path=temp_path)

    ### FIX FROM HERE ###
    updated_speaker_mapping = link_unified_mappings(diarized_combined, speaker_mapping)

    # update speaker mappings based on word-level transcript
    update_transcripts(speaker_map=updated_speaker_mapping)

    # update the library 
    update_speaker_library(updated_speaker_mapping, utterance_data)

    # append to the queriable transcript
    append_to_transcript(word_diarized_path,timestamp_dt,transcript_file_location)


def transcribe_audio_chunk(input_path, timestamp, input_containing_path=RAW_PATH, processed_folder_path=PROCESSED_PATH):

    # STEP 1: save the word-level transcript to json
    word_level_transcript, transcript = diarize_file(input_path, input_containing_path, processed_folder_path)

    # STEP 2: link the individuals to the library, correct the json transcript
    process_recording(word_level_transcript, timestamp)

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1].lower() == "example":
            file_name = "mango_kefir_smoothie.wav"
            timestamp_dt = datetime.now()
        else:
            file_name, timestamp_dt = record_audio_chunk()
        
        transcribe_audio_chunk(input_path=file_name, timestamp=timestamp_dt, input_containing_path=RAW_PATH, processed_folder_path=PROCESSED_PATH)
    
    except KeyboardInterrupt:
        print("Recording stopped.")

