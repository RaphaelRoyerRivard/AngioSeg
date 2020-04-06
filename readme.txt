For AngioSeg with tensorflow 1.3: at last an easy installation
- First install CUDA toolkit 8.0
- Unzip Cudnn v6.0 for cuda 8.0  + add <installpath>\bin to the PATH environment variable.
- install python 3.5.2 64 bits in the preferred location (please note only one version of python 3 can be in the path) or anaconda 3 64 bits to create specific environment
- then open command pip3 install --upgrade tensorflow-gpu
- cd in Package directory
- install numpy with pip install numpy-1.13.3+mkl-cp35-cp35m-win_amd64.whl
- install scipy with pip install scipy-1.0.0rc1-cp35-cp35m-win_amd64.whl
- install opencv from the wheel provided by http://www.lfd.uci.edu/~gohlke/pythonlibs/ pip install opencv_python-3.3.0+contrib-cp35-cp35m-win_amd64.whl
- install PyQt4 with pip install PyQt4-4.11.4-cp35-cp35m-win_amd64.whl
- install Pillow with pip install Pillow-4.3.0-cp35-cp35m-win_amd64.whl
- install matplotlib with pip install matplotlib-2.1.0-cp35-cp35m-win_amd64.whl
- install TortoiseSVN
- install Pycharm or favorite IDE
- checkout https://192.168.0.107/svn/AngioSeg/trunk
- voir python_dependencies pour toutes les librairies et leurs versions au moment du test. (normalement on peut mettre à jour toutes les librairies sans impact sur le code - tensorflow change de moins en moins son API)

- Apres installation de Visual, installer Nsight pour avoir la compil CUDA

- les 3 modèles sont ici: E:\Fantin\AngioData\checkpoint
checkpointimageback = modele avec image background 1024x1024
checkpointimage = modele avec image 1024x1024 (utilisé notamment dans l'interface SEGMENTATIONTOOL)
checkpointimageback512 = modele avec image background 512x512 (utilisé avec VIDEOTOOL)


-avant de lancer VIDEOTOOL il faut ouvrir visual studio et compiler les solutions suivantes en Release x64 : src\VESSELANALYSIS\CUDASkel\CUDASkel.sln et src\VESSELANALYSIS\VesselAnalysis\VesselAnalysis.sln

Pour lancer SEGMENTATIONTOOL --> labelAV.py
Pour lancer VIDEOTOOL --> camerawidget.py
