from MIDI_Retrieval_System import BootlegScore, MIDIProcessing

if __name__ == "__main__":

    # random example
    midifile = './data/midi/p91.mid'
   
    # visualize bootleg score
    midi = MIDIProcessing(midifile)
    result, _ =  midi.getNoteEvents(quant=10)
    both, times, numNotes, _, _ = midi.generateBootlegScore(result, 2, 2)
    bs = BootlegScore(both[:,0:140])
    bs.visualize(staff_lines=MIDIProcessing.staffLinesBoth)

    # process midi batch
    fileList = './cfg_files/midi.train.list' # list of all midi files to process
    outdir = 'experiments/train/db' # where to save bootleg scores
    MIDIProcessing.processMidiBatch(fileList, outdir)
    
