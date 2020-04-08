@echo off
if "%1" == "" (
    echo Path is not defined
    exit /B
)
echo Segmenting and skeletonizing images in subtree of path %1
set script_path=%~dp0
set project_path=%script_path:~0,-8%
set segmentationtool_path=%project_path%\src\SEGMENTATIONTOOL\label_AV.py
set vesselanalysis_path=%project_path%\src\VESSELANALYSIS\vesselanalysis.py
echo %segmentationtool_path%
echo %vesselanalysis_path%
::python %segmentationtool_path% --input %1
python %vesselanalysis_path% --input %1
