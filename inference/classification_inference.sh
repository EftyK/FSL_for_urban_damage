#!/bin/bash 

#####################################################################################################################################################################
# Code adapted from https://github.com/DIUx-xView/xView2_baseline/blob/master/utils/inference.sh

# xview2-baseline Copyright 2019 Carnegie Mellon University. BSD-3

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution. 
# Please see Copyright notice for non-US Government use and distribution.                                                                                                                                                #
#####################################################################################################################################################################

set -euo pipefail

# this function is called when Ctrl-C is sent
function trap_ctrlc ()
{
    # perform cleanup here
    echo "Ctrl-C or Error caught...performing clean up check /tmp/inference.log"

    if [ -d /tmp/inference ]; then
           rm -rf /tmp/inference
    fi

    exit 99
}

# initialise trap to call trap_ctrlc function
# when signal 2 (SIGINT) is received
trap "trap_ctrlc" 2 9 13 3

help_message () {
        printf "${0}: Runs the polygonization in inference mode\n\t-x: path to test data\n\t-i: /full/path/to/input/pre-disaster/image.png\n\t-p: /full/path/to/input/post-disaster/image.png\n\t-o: /path/to/output.png\n\t-l: path/to/localization_weights\n\t-c: path/to/classification_weights\n\t-e /path/to/virtual/env/activate\n\t-y continue with local environment and without interactive prompt\n\n"
}

input=""
input_post=""
inference_base="/tmp/inference"
LOGFILE="/tmp/inference_log"
XBDIR=""
virtual_env=""
localization_weights=""
classification_weights=""
continue_answer="n"

if [ "$#" -lt 13 ]; then
        help_message
        exit 1 
fi 

while getopts "i:p:o:x:l:e:c:hy" OPTION
do
     case $OPTION in
         h)
             help_message
             exit 0
             ;;
         y)
             continue_answer="y"
             ;;
         o)
             output_file="$OPTARG"
             ;;
         x)
             DATADIR="$OPTARG"
             ;;
         i)
             input="$OPTARG"
             ;;
         p)
             input_post="$OPTARG"
             ;;
         l)
             localization_json="$OPTARG"
             ;;
         c)
             classification_weights="$OPTARG"
             ;;
         e)
             virtual_env="$OPTARG"
             ;;
         ?)
             help_message
             exit 0
             ;;
     esac
done

# Create the output directory if it doesn't exist 
mkdir -p "$inference_base"

if ! [ -f "$LOGFILE" ]; then
    touch "$LOGFILE"
fi

printf "==========\n" >> "$LOGFILE"
echo `date +%Y%m%dT%H%M%S` >> "$LOGFILE"
printf "\n" >> "$LOGFILE"

input_image=${input##*/}

label_temp="$inference_base"/"${input_image%.*}"/labels
mkdir -p "$label_temp"

printf "\n"

printf "\n"

# Run in inference mode
# Because of the models _have_ to be in the correct directory, they use relative paths to find the source (e.g. "../src") 
# sourcing the virtual environment packages if they exist
# this is *necessary* or all packages must be installed globally
if [ -f  "$virtual_env" ]; then
    source "$virtual_env"
else
    if [ "$continue_answer" = "n" ]; then 
        printf "Error: cannot source virtual environment  \n\tDo you have all the dependencies installed and want to continue? [Y/N]: "
        read continue_answer 
        if [ "$continue_answer" == "N" ]; then 
               exit 2
        fi 
    fi
fi

# Classification inferences start below
# Replace the pre image here with the post
# We need to do this so the classification inference pulls the images from the post 
# Since post is where the damage occurs
printf "Grabbing post image file for classification\n"
disaster_post_file="$input_post"

mkdir -p "$inference_base"/output_polygons

printf "Running classification\n" 

# Extracting polygons from post image 
python3 ./process_data_inference.py --input_img "$disaster_post_file" --label_path "$localization_json" --output_dir "$inference_base"/output_polygons --output_csv "$inference_base"/output.csv >> "$LOGFILE" 2>&1

# Classifying extracted polygons 
python3 ./inference_proto.py --test_data "$DATADIR"/ --test_csv "$DATADIR"/train.csv --model_weights "$classification_weights" --output_json /tmp/inference/classification_inference.json >> "$LOGFILE" 2>&1


printf "\n" >> "$LOGFILE"

# Combining the predicted polygons with the predicted labels, based off a UUID generated during the localization inference stage  
printf "Formatting json and scoring image\n"
python3 combine_jsons.py --polys "$localization_json" --classes /tmp/inference/classification_inference.json --output "$inference_base/inference.json" >> "$LOGFILE" 2>&1
printf "\n" >> "$LOGFILE"

# Transforming the inference json file to the image required for scoring
printf "Finalizing output file" 
python3 inference_image_output.py --input "$inference_base"/inference.json --output "$output_file"  >> "$LOGFILE" 2>&1


#Cleaning up by removing the temporary working directory we created
printf "Cleaning up\n"
# rm -rf "$inference_base"

printf "==========\n" >> "$LOGFILE"
printf "Done!\n"

