import fnmatch
import os
import random
import re
import threading

import librosa
import numpy as np
import tensorflow as tf

CATEGORY_FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'
FILE_PATTERN = r'(?:\d\d-){3}[a-zA-Z]-(\d{3})-[a-zA-Z]{3}-[a-zA-Z]{2}\.wav'


def get_category_cardinality(files):
    id_reg_expression = re.compile(CATEGORY_FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        id, recording_id = [int(id_) for id_ in matches]
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id


def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern and with the specified language encoded in the filename.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        # TODO: filter by lang
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate, input_lang, output_lang):
    # TODO: Modify this method to load input and output files in tandem
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    id_reg_exp = re.compile(CATEGORY_FILE_PATTERN)
    speech_id_reg_exp = re.compile(FILE_PATTERN)
    input_files = [file for file in files if 'org-' + input_lang in file]
    print("input files length: {}".format(len(input_files)))
    randomized_files = randomize_files(input_files)
    for input_filename in randomized_files:
        ids = id_reg_exp.findall(input_filename)
        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            category_id = None
        else:
            # The file name matches the pattern for containing ids.
            category_id = int(ids[0][0])
        ids = speech_id_reg_exp.findall(input_filename)
        print("ids: " + str(ids))
        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            speech_id = None
        else:
            # The file name matches the pattern for containing ids.
            speech_id = ids[0][0]
        input_audio, _ = librosa.load(input_filename, sr=sample_rate, mono=True)
        input_audio = input_audio.reshape(-1, 1)

        # Load the corresponding output file
        output_files = [file for file in files if speech_id + '-' + 'int-' + input_lang + '-' + output_lang in file]
        output_filename = output_files[0]
        output_audio, _ = librosa.load(output_filename, sr=sample_rate, mono=True)
        output_audio = output_audio.reshape(-1, 1)

        print("Input and output audio files loaded successfully")
        yield input_audio, input_filename, output_audio, output_filename, category_id


def trim_silence(audio, threshold, frame_length=2048):
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rmse(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def not_all_have_id(files):
    ''' Return true iff any of the filenames does not conform to the pattern
        we require for determining the category id.'''
    id_reg_exp = re.compile(CATEGORY_FILE_PATTERN)
    for file in files:
        ids = id_reg_exp.findall(file)
        if not ids:
            return True
    return False


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 coord,
                 sample_rate,
                 gc_enabled,
                 receptive_field,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=32,
                 input_lang='es',
                 output_lang='en'):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.receptive_field = receptive_field
        self.silence_threshold = silence_threshold
        self.gc_enabled = gc_enabled
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])
        self.input_lang=input_lang
        self.output_lang=output_lang

        if self.gc_enabled:
            self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=())
            self.gc_queue = tf.PaddingFIFOQueue(queue_size, ['int32'],
                                                shapes=[()])
            self.gc_enqueue = self.gc_queue.enqueue([self.id_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        files = find_files(audio_dir)
        if not files:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))
        if self.gc_enabled and not_all_have_id(files):
            raise ValueError("Global conditioning is enabled, but file names "
                             "do not conform to pattern having id.")
        # Determine the number of mutually-exclusive categories we will
        # accomodate in our embedding table.
        if self.gc_enabled:
            _, self.gc_category_cardinality = get_category_cardinality(files)
            # Add one to the largest index to get the number of categories,
            # since tf.nn.embedding_lookup expects zero-indexing. This
            # means one or more at the bottom correspond to unused entries
            # in the embedding lookup table. But that's a small waste of memory
            # to keep the code simpler, and preserves correspondance between
            # the id one specifies when generating, and the ids in the
            # file names.
            self.gc_category_cardinality += 1
            print("Detected --gc_cardinality={}".format(
                  self.gc_category_cardinality))
        else:
            self.gc_category_cardinality = None

    def dequeue(self, num_elements):
        # TODO: dequeue the output as well as the input
        output = self.queue.dequeue_many(num_elements)
        return output

    def dequeue_gc(self, num_elements):
        # TODO: dequeue the output as well as the input
        return self.gc_queue.dequeue_many(num_elements)

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_generic_audio(self.audio_dir, self.sample_rate, self.input_lang, self.output_lang)
            for input_audio, input_filename, output_audio, output_filname, category_id in iterator:
                # TODO: Use output_audio
                if self.coord.should_stop():
                    stop = True
                    break
                if self.silence_threshold is not None:
                    # Remove silence
                    input_audio = trim_silence(input_audio[:, 0], self.silence_threshold)
                    input_audio = input_audio.reshape(-1, 1)
                    if input_audio.size == 0:
                        print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the audio."
                              .format(input_filename))

                input_audio = np.pad(input_audio, [[self.receptive_field, 0], [0, 0]],
                               'constant')

                if self.sample_size:
                    # Cut samples into pieces of size receptive_field +
                    # sample_size with receptive_field overlap
                    # TODO: I think this is where we need to enqueue the output along with the input
                    # TODO: Make sure the the input audio and output audio have the same number of samples
                    while len(input_audio) > self.receptive_field:
                        piece = input_audio[:(self.receptive_field +
                                        self.sample_size), :]
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        input_audio = input_audio[self.sample_size:, :]
                        if self.gc_enabled:
                            sess.run(self.gc_enqueue, feed_dict={
                                self.id_placeholder: category_id})
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: input_audio})
                    if self.gc_enabled:
                        sess.run(self.gc_enqueue,
                                 feed_dict={self.id_placeholder: category_id})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
