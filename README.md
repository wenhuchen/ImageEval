# ImageEditing
Editing Baselines


## For textual inversion

```
cd textual_inversion
accelerate run --files [YOUR_DIR] --special_token [TOKEN] --initialize [WORD]
accelrate run --files dogs/ --special_token [dog] --initialize dog
```

## For Null-Text Inversion

```
cd prompt-to-prompt
```
open null_text_w_ptp.ipynb, follow the instruction to do image editing
