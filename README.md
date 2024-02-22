# Welcome to PyDistort

PyDistort is framework to train LSTM networks that can add FX to a clean (or rather any) audio signal. It creates patches given a clean and a patched signal for training

It started out as project to model guitar amps but should be able to add any effect to any input signal. 

PyDistort is currently under development

## Limitations

**Dataset noise** : If there is noise in the input or output signal in training data, it will also be replicated in the end model.

**File length** : For simplicity, it works second by second. Hence the last milliseconds will be truncated before processing. Make sure to add enough padding when predicting.

**File Format** : Only supports 44.1kHz .wav files....at the moment

## Future and Contribution

I plan to provide a GA based Hyperparam tuning. Probably overkill but Its for my learning

I do plan to provide this as a package that can be trained for customer patches. Although LSTMs take too long to train. So Good Luck.

Contribution : Submit your PRs and I'll merge them if I like it, Or feel free to fork the repo. 