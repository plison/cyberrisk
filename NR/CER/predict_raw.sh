python convert_raw2tabular.py --infile $1 --outfile sample_raw_doc.conllu
python run.py predict --fdata sample_raw_doc.conllu --model casie-cer.model --vocab casie-cer.vocab --device -1 --fpred casie-cer-raw-pred.conllu
python extract_entities.py --labelled_data casie-cer-raw-pred.conllu --outfile $2
rm sample_raw_doc.conllu
rm casie-cer-raw-pred.conllu
