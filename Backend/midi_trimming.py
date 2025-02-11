import mido

def trim_midi(input_path, output_path, start_time_ms, end_time_ms):
    mid = mido.MidiFile(input_path)
    ticks_per_beat = mid.ticks_per_beat
    tempo = 500000  # Default tempo (500,000 µs per beat)

    # Get actual tempo from the MIDI file
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo  # Update tempo from the file
                break

    # Convert milliseconds to MIDI ticks
    start_tick = mido.second2tick(start_time_ms, ticks_per_beat, tempo)
    end_tick = mido.second2tick(end_time_ms, ticks_per_beat, tempo)

    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    for track in mid.tracks:
        new_track = mido.MidiTrack()
        new_midi.tracks.append(new_track)

        time_elapsed = 0

        for msg in track:
            time_elapsed += msg.time  # Track elapsed time in ticks

            if start_tick <= time_elapsed <= end_tick:
                new_track.append(msg)

            elif time_elapsed > end_tick:
                break  # Stop processing once past end time

        if len(new_track) > 0:
            new_track.insert(0, mido.MetaMessage('set_tempo', tempo=tempo))  # Add tempo

    if any(len(track) > 1 for track in new_midi.tracks):
        new_midi.save(output_path)
    else:
        print("Error: Trimmed MIDI is empty!")
    