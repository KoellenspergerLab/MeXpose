# Aliases for app launch commands within the container
alias fiji-gui="home/Fiji.app/ImageJ-linux64"
alias cellpose="python -m cellpose"
alias cellprofiler-gui="python -m cellprofiler"
alias cellprofiler="python -m cellprofiler -c -r --plugins-directory=/home/plugins"
alias analysis-gui="jupyter-lab --config=/home/MeXpose/jupyter_lab_config.py /home/MeXpose/scripts/analysis_notebook.ipynb --allow-root"
alias quantify-gui="jupyter-lab --config=/home/MeXpose/jupyter_lab_config.py /home/MeXpose/scripts/quantification_notebook.ipynb --allow-root"
alias pixel-gui="jupyter-lab --config=/home/MeXpose/jupyter_lab_config.py /home/MeXpose/scripts/pixel_notebook.ipynb --allow-root"