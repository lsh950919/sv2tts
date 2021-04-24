from multiprocessing.pool import Pool
from synthesizer import audio
from functools import partial
from itertools import chain
from encoder import inference as encoder
from pathlib import Path
from utils import logmmse
from tqdm import tqdm
import numpy as np
import librosa
import os
##


def preprocess_KSponSpeech(datasets_root: Path, out_dir: Path, n_processes: int,
                           skip_existing: bool, hparams):
    # Gather the input directories
    dataset_root = datasets_root.joinpath("KSponSpeech")
    input_dirs = [dataset_root.joinpath("KsponSpeech_01"),
    dataset_root.joinpath("KsponSpeech_02"),
    dataset_root.joinpath("KsponSpeech_03"),
    dataset_root.joinpath("KsponSpeech_04"),
    dataset_root.joinpath("KsponSpeech_05")
    #datasets_root.joinpath('test')
    ]

    print("\n    ".join(map(str, ["Using data from:"] + input_dirs)))
    assert all(input_dir.exists() for input_dir in input_dirs)

    # Create the output directories for each output file type
    out_dir.joinpath("mels").mkdir(exist_ok=True)
    out_dir.joinpath("audio").mkdir(exist_ok=True)

    # Create a metadata file
    metadata_fpath = out_dir.joinpath("train.txt")
    metadata_file = metadata_fpath.open("a" if skip_existing else "w", encoding="cp949")

    # Preprocess the dataset
    speaker_dirs = list(chain.from_iterable(input_dir.glob("*") for input_dir in input_dirs))  # 폴더안의 모든 폴더(Speaker)
    func = partial(preprocess_speaker, out_dir=out_dir, skip_existing=skip_existing,
                   hparams=hparams)
    job = Pool(n_processes).imap(func, speaker_dirs)
    for speaker_metadata in tqdm(job, "KSponSpeech", len(speaker_dirs), unit="speakers"):
        for metadatum in speaker_metadata:
            metadata_file.write("|".join(str(x) for x in metadatum) + "\n")
    metadata_file.close()

    # Verify the contents of the metadata file
    with metadata_fpath.open("r", encoding="cp949") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
    mel_frames = sum([int(m[4]) for m in metadata])
    timesteps = sum([int(m[3]) for m in metadata])
    sample_rate = hparams.sample_rate
    hours = (timesteps / sample_rate) / 3600
    print("The dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
          (len(metadata), mel_frames, timesteps, hours))
    print("Max input length (text chars): %d" % max(len(m[5]) for m in metadata))
    print("Max mel frames length: %d" % max(int(m[4]) for m in metadata))
    print("Max audio timesteps length: %d" % max(int(m[3]) for m in metadata))


def preprocess_speaker(speaker_dir, out_dir: Path, skip_existing: bool, hparams):
    metadata = []
    check_list = [",01", ",02", ",03", ",04", ",05", ",06", ",07", ",08", ",09"]
    # Gather the utterance audios and texts
    # try:
    files = os.listdir(speaker_dir)

    for file in files:
        if file.endswith("alignment.txt"):
            with open(os.path.join(speaker_dir, file), "r", encoding='cp949') as alignments_file:
                alignments = [line.rstrip().split(" ") for line in alignments_file.readlines()]
    # # except StopIteration:
    # #     # A few alignment files will be missing
    # #     continue

    # Iterate over each entry in the alignments file
    for wav_fname, words in alignments:

        for check in check_list:
            if check in words:
                print(words)
                words = "pass"

        wav_fpath = speaker_dir.joinpath(wav_fname + ".pcm")
        assert wav_fpath.exists()
        # words = words.replace("\"", "").split(",")
        # end_times = list(map(float, end_times.replace("\"", "").split(",")))
        #
        # # Process each sub-utterance
        wavs = normalization(wav_fpath, hparams)

        if wavs is not None and words is not "pass":
            sub_basename = "%s" % (wav_fname)
            metadata.append(process_utterance(wavs, words, out_dir, sub_basename,
                                              skip_existing, hparams))

    return [m for m in metadata if m is not None]


def normalization(wav_fpath, hparams):
    try:
        wav = np.memmap(wav_fpath, dtype='h', mode='r')
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max
    except EOFError:
        print(wav_fpath)
        return None
    return wav


def process_utterance(wav: np.ndarray, text: str, out_dir: Path, basename: str,
                      skip_existing: bool, hparams):
    ## FOR REFERENCE:
    # For you not to lose your head if you ever wish to change things here or implement your own
    # synthesizer.
    # - Both the audios and the mel spectrograms are saved as numpy arrays
    # - There is no processing done to the audios that will be saved to disk beyond volume
    #   normalization (in split_on_silences)
    # - However, pre-emphasis is applied to the audios before computing the mel spectrogram. This
    #   is why we re-apply it on the audio on the side of the vocoder.
    # - Librosa pads the waveform before computing the mel spectrogram. Here, the waveform is saved
    #   without extra padding. This means that you won't have an exact relation between the length
    #   of the wav and of the mel spectrogram. See the vocoder data loader.

    # Skip existing utterances if needed
    mel_fpath = out_dir.joinpath("mels", "mel-%s.npy" % basename)
    wav_fpath = out_dir.joinpath("audio", "audio-%s.npy" % basename)
    if skip_existing and mel_fpath.exists() and wav_fpath.exists():
        return None

    # Skip utterances that are too short
    if len(wav) < hparams.utterance_min_duration * hparams.sample_rate:
        return None

    # Compute the mel spectrogram
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    # Skip utterances that are too long
    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None

    # Write the spectrogram, embed and audio to disk
    np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    np.save(wav_fpath, wav, allow_pickle=False)

    # Return a tuple describing this training example
    return wav_fpath.name, mel_fpath.name, "embed-%s.npy" % basename, len(wav), mel_frames, text


def embed_utterance(fpaths, encoder_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    # Compute the speaker embedding of the utterance
    wav_fpath, embed_fpath = fpaths
    wav = np.load(wav_fpath)
    wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(wav)
    np.save(embed_fpath, embed, allow_pickle=False)


def create_embeddings(synthesizer_root: Path, encoder_model_fpath: Path, n_processes: int):
    wav_dir = synthesizer_root.joinpath("audio")
    metadata_fpath = synthesizer_root.joinpath("train.txt")
    assert wav_dir.exists() and metadata_fpath.exists()
    embed_dir = synthesizer_root.joinpath("embeds")
    embed_dir.mkdir(exist_ok=True)

    # Gather the input wave filepath and the target output embed filepath
    with metadata_fpath.open("r") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
        fpaths = [(wav_dir.joinpath(m[0]), embed_dir.joinpath(m[2])) for m in metadata]

    # TODO: improve on the multiprocessing, it's terrible. Disk I/O is the bottleneck here.
    # Embed the utterances in separate threads
    func = partial(embed_utterance, encoder_model_fpath=encoder_model_fpath)
    job = Pool(n_processes).imap(func, fpaths)
    list(tqdm(job, "Embedding", len(fpaths), unit="utterances"))
