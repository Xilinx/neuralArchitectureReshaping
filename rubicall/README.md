# Reproducing The Results in The Paper

## Prerequisites

We list the links to download and compile each tool for comparison below:

* [Minimap2 (v2.24)](https://github.com/lh3/minimap2/releases/tag/v2.24)
* [MHAP (v2.1.3 -- via conda)](https://anaconda.org/bioconda/mhap/2.1.3/download/noarch/mhap-2.1.3-hdfd78af_1.tar.bz2)
* [LRA (v1.3.2 -- via conda)](https://anaconda.org/bioconda/lra/1.3.2/download/linux-64/lra-1.3.2-ha140323_0.tar.bz2)
* [Winnowmap (v2.03 -- via conda)](https://anaconda.org/bioconda/winnowmap/2.03/download/linux-64/winnowmap-2.03-h2e03b76_0.tar.bz2)
* [S-conLSH (v2.0)](https://github.com/anganachakraborty/S-conLSH-2.0/tree/292fbe0405f10b3ab63fc3a86cba2807597b582e)

We use various tools to process and analyze the data we generate using each tool. The following tools must also be installed in your machine:

* [SAMtools](https://github.com/samtools/samtools/releases/download/1.14/samtools-1.14.tar.bz2)
* [miniasm](https://github.com/lh3/miniasm/tree/ce615d1d6b8678d38f2f9d27c9dccd944436ae75)
* [MUMmer v4.0.0rc1](https://github.com/mummer4/mummer/releases/tag/v4.0.0rc1)
* [Seqtk](https://github.com/lh3/seqtk)
* [BamUtil v1.0.15](https://github.com/statgen/bamUtil/releases/tag/v1.0.15)
* [bedtools v2.30.0](https://github.com/arq5x/bedtools2/releases/tag/v2.30.0)
* [mosdepth v0.3.2](https://github.com/brentp/mosdepth/releases/tag/v0.3.2)
* [QUAST v5.0.2](https://github.com/ablab/quast/releases/tag/quast_5.0.2)
* [PBSIM2 v2.0.1 -- via conda](https://anaconda.org/bioconda/pbsim2/2.0.1/download/linux-64/pbsim2-2.0.1-h9f5acd7_1.tar.bz2)
* [Inspector v1.2](https://github.com/ChongLab/Inspector)
* [BBMap v39.03](https://sourceforge.net/projects/bbmap/)
* [SeqKit  v2.5.1](https://github.com/shenwei356/seqkit)

We suggest using Conda to install these tools with their specified versions, as almost all of them are included in the Conda repository.

Please make sure that all of these tools are in your `PATH`


## Datasets
Make sure to download the ONT official training dataset using the following:

``` bash
$ rubicon download --training # will download ONT dataset 
```
Evaluated species reads can be downloaded using `rubicon download --organism`. The datasets will be downloaded under `~/rubicon/rubicon/data/organism`

Evaluated reference files can be downloaded from https://bridges.monash.edu/articles/dataset/Reference_genomes/10198815

For the human genome, we download reads from https://labs.epi2me.io/gm24385_2020.11/, while the reference genome is available at https://github.com/marbl/HG002.

We also provide references under `reproducibleEvaluation/reference_files`

``` bash
$ cd reference_files
$ bash download_reference.sh
```

Trained models are available under `reproducibleEvaluation/models`

``` bash
$ cd models
$ bash download_bonito.sh
$ bash download_crf-fast.sh
$ bash download_crf-sup.sh
$ bash download_rubiconqabas.sh
$ bash download_staticquant.sh
$ bash download_rubicallmp.sh
$ bash download_rubiconnoskipfp.sh
$ bash download_dorado.sh
$ bash download_sacall.sh
# Causalcall can be downloaded from their official git repo https://github.com/scutbioinformatic/causalcall
```

Now that you have downloaded all the datasets, we can start running all the tools to collect the results.



## Training a Model From Scratch


Train a model from scratch.

```bash
rubicon train path_to_store_trained_model -f --full --batch 64 --no-amp --epoch 100  --temp 2 --alpha 0.9 --type rubiconqabas-mp --teacher
```


`--type` type of model to use `{rubiconqabas-mp,bonito, staticquant}`. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Use `rubiconqabas-mp` to train QABAS model.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Use `bonito` to train our Bonito_CTC model.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Use `bonitostaticquant` to train models that are statically quantized.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To change weights and activation precision: Update `layer_weight_bits` and `act_bits` variables in `rubicon/arch/bonitostaticquant.py`

`--teacher` flag will perform training using knowledge distillation (KD). 

`--temp` temperature for KD (default=2).

`--alpha` alpha for KD (default=0.9).

`-struct` passing enables structured channel pruning, else unstructed element pruning.

`-l1` choose pruning weights using L1-norm, else weights are randomly selected.

We ran QABAS using the following command to get `rubiconqabas-mp` for 96 hours.

```bash
rubicon qabas path_to_store_search_data -f --chunks=30000 --batch 64 --no-amp --epoch 100 --reference_latency 7500 --applied_hardware aie_lut --nas proxy --rub_ctrl_opt --rub_arch_opt --rub_sched --default
```

## SkipClip with Knowledge Distillation
Once `rubiconqabas-mp` model is trained, then apply SkipClip to achieve 'RUBICALL-MP'

```bash
rubicon skipclip path_to_store_rubicall-mp  --pretrained path_to_trained_rubiconqabas-mp -f --full --batch 64 --no-amp --epoch 100  --temp 2 --alpha 0.9 --type rubiconqabas-mp
```

`--teacher_directory` add teacher path (`default=models/bonito`).

`--full` full the complete dataset.

`--temp` temperature for KD (default=2).

`--alpha` alpha for KD (default=0.9).

`--type` performs SkipClip on QABAS generated model.

`--skip_stride` changes the stride at which the skip connection should be removed.

```bash
rubicon skipclip path_to_store_rubicall-mp  --pretrained reproducibleEvaluation/models/rubiconqabas/ -f --full --batch 64 --no-amp --epoch 100  --temp 2 --alpha 0.9 --type rubiconqabas
```

## Basecalling a Specie
Download our generated basecalled reads

``` bash
$ cd models
$ bash download_reads_bonito.sh
$ bash download_reads_crf-fast.sh
$ bash download_reads_crf-sup.sh
$ bash download_reads_causalcall.sh
$ bash download_reads_dorado.sh
$ bash download_reads_rubicall.sh
$ bash download_reads_sacall.sh
```

Or perform basecalling on a set of raw reads. Make sure models are available under `reproducibleEvaluation/models` and evaluated organisms under `../../data/organism`

```bash
cd reproducibleEvaluation/basecalled_reads
python run_basecalling.py specie_dir model_dir modeltype
```

`specie_dir` is the path to specie reads (Can be downloaded using `rubicon download --organism`) 

`model_dir` is the path to a specific model, which are under reproducibleEvaluation/models

`modeltype` depending upon the model

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`crf-fast` for bonito_crf-fast model

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`crf-sup` for bonito_crf-sup model

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`bonito` for bonito_ctc model

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`rubiconqabas` for model after QABAS

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`rubiconskiptrim` for model after RUBICALL-MP

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`rubiconnoskipfp` for model after RUBICALL-FP

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`sacall` for bonito_ctc model

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`staticquant` for model after statically quant models

To run all the models that we evaluate, simply run the command below. 

```bash
cd reproducibleEvaluation
python run_all.py
```


## Basecalling Accuracy Analysis
For basecalling accuracy, we align each basecalled read to its corresponding reference genome of the same species using the state-of-the-art read mapper, minimap2. We use [basecalling-comparison scripts](https://github.com/rrwick/Basecalling-comparison).


```bash
cd reproducibleEvaluation/basecalling_accuracy_analysis
```

Note we do not perform nanopolishing in `~/reproducibleEvaluation/basecalling_accuracy_analysis/basecalling_comparison/analysis_scripts/analysis.sh`

Make sure reference files are in the same folder from where you execute the script as reference$i.fasta. 
Where $i is the name of the species (e.g., reference_Acinetobacter_pittii_16-377-0801.fasta)

``` bash
cd reproducibleEvaluation/basecalling_accuracy_analysis
$ bash download_reference.sh
```

Appropriate reference files are under `reproducibleEvaluation/reference_files`
The human reference genome v0.7 is available at https://github.com/marbl/HG002

```bash
#run the analysis using the script below
python reproducibleEvaluation/basecalling_accuracy_analysis/run_eval_pipeline.py path_to_basecalled_reads
```
To analyze the results, run the following command. Pass the path to generated _read.tsv file.

```bash
python3 analysis_scripts/get_median_identity.py <generated_basecalled_reads.tsv>
```

## Downstream Analysis: De novo Assembly 
We construct de novo assemblies from the basecalled reads and calculate the statistics related to the accuracy, completeness, and contiguity of these assemblies. To generate de novo assemblies, we use minimap2  to report all read overlaps and miniasm to construct the assembly from these overlaps. We use miniasm because it allows us to observe the effect of the reads on the assemblies without performing additional error correction steps on input reads and their final assembly. To measure the assembly accuracy, we use dnadiff to evaluate 1) the portion of the reference genome that can align to a given assembly (i.e., Genome Fraction) and 2) the average identity of assemblies (i.e., Average Identity) when compared to their respective reference genomes. To measure statistics related to the contiguity and completeness of the assemblies, such as the overall assembly length, average GC content (i.e., the ratio of G and C bases in an assembly), and NG50 statistics (i.e., shortest contig at the half of the overall reference genome length), we use QUAST. 

```bash
cd reproducibleEvaluation/denovo_assembly/
bash download_analysis_data.sh #download necessary data
#Run assembly for each basecaller. E.g., below, we run for bonito. In the same way, run for rubicall, crf-fast, causalcall.
cd reproducibleEvaluation/denovo_assembly/bonito
./run_assembly.sh 
```

Run dnadiff and quast:

```bash
cd  reproducibleEvaluation/denovo_assembly/analysis
#Runs dnadiff and quast to generate all the results regarding the assembly and overlap statistics. 
./run.sh
```

Similarly, for the human genome, follow the below instructions. Please make sure run_dnadiff and run_quast, points to proper location:

```bash
cd reproducibleEvaluation/denovo_assembly/human_analysis
./run_assembly.sh 
```

It outputs the result at the end to the standard output.



## Downstream Analysis: Read Mapping
We basecall the raw electrical signals into reads using each of the subject basecallers. We map the resulting read set to its reference genome of the same species using the state-of-the-art read mapper, minimap2. We use the default parameter values for mapping ONT reads using the preset parameter -x map-ont. We use the stats tool from the SAMtools library to obtain four key statistics on the quality of read mapping results: the total number of mismatches, the total number of mapped bases, the total number of mapped reads, and the total number of unmapped reads.


```bash
cd reproducibleEvaluation/read_mapping_analysis
python run_minimap.py ref_path fasta_path sam_out_path
#run minimap
#ref_path is the path of the reference (references available under reproducibleEvaluation/reference_files)
#fasta_path is the path to basecalled read (generated basecalled reads available under: reproducibleEvaluation/basecalled_reads)
#sam_out_path path to store files needed for sam tools
```
Run samtool on generated files. 

```bash
cd reproducibleEvaluation/read_mapping_analysis
python run_samtools.py sam_dir_path out_dir_path
#run samtools
#sam_dir_path path to store files needed for sam tools
#out_dir_path path to store samtool results
```
Read stats for mismatches, bases mapped, reads mapped, and reads unmapped
```bash
cd reproducibleEvaluation/read_mapping_analysis
python read_stats.py
```

Download human genome analysis:

```bash
cd reproducibleEvaluation/read_mapping_analysis
bash download_human_samtools_results.sh
```


## K-mer Counting Analysis
We analyze the occurrence of k-mer (i.e., substrings of length k) in a given sequence of basecalled reads and their assemblies. We use BBMap to collect the number of unique k-mers and the frequency of each unique k-mer in a given sequence. During our analysis, we vary the value of k from 15 to 31. Based on our empirical analysis, we set the k value for our evaluated bacterial species to 15, where we observe distinct peaks of unique k-mers.

To collect distinct k-mers (as sequences) and the number of times each distinct k-mer appears, run the following on generated reads and assembly with proper paths for BBMap, Reads/Assembly, and output path

```bash
run_k-mer_unique_reads.py
run_k-mer_unique_assembly.py
```

To generate a k-mer frequency histogram (used to find peaks and threshold to generate over and under-represented k-mers), run the following on generated reads and assembly with proper paths for BBMap, Reads/Assembly, and output path

```bash
run_hist_compare_reads.py
run_hist_compare_assembly.py
```

 Generate the k-mer list that appears less than a certain threshold:

```bash
run_under.sh <threshold> <k-mer-file for read/assembly>
For example:
run_under.sh 10 k-mer_compare_15/basecalled_read/bonito/serratia_marcescens_17.out # setting threshold=10 for serratia_marcescens_17 reads generated using bonito basecaller
```
 Generate the k-mer list that appears more than a certain threshold:

```bash
run_over.sh <threshold> <k-mer-file for read/assembly>
For example:
run_over.sh 10 k-mer_compare_15/basecalled_read/bonito/serratia_marcescens_17.out # setting threshold=10 for serratia_marcescens_17 reads generated using bonito basecaller
```

Find the intersection of under and over-represented k-mers between reads and assembly using the following script:

```bash
run_intersect_k-mers.sh A.over B.over common_kmer.over
# A from read set, B from assembly for the same dataset and basecalling
```
Run the above script for both .over and .under for each dataset and basecalling pairs. Then count the number of lines in common_k_mer.over. Use the common count to take its ratio with either read set and assembly for the over/under k-mers from the same basecalling+dataset.

## ONNX
We provide the generated ONNX (Open Neural NetworkExchange) file for RUBICALL_MP.

We estimate the performance of AIE-ML by calculating bit operations (BOPs) that measure the number of bitwise operations in a given network from its ONNX representation and consider the total number of supported operations per datatype on AIE-ML.

## Figures
You will find the high-quality figures that we included in the paper. We also provide scripts to generate these figures.
