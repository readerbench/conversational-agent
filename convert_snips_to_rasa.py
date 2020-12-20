import re
import sys
import yaml

def convert_snips_to_rasa(path=sys.argv[1]):
	with open(path, 'r') as yamlfile:
		for obj in yaml.load_all(yamlfile.read()):
			if obj['type'] != 'intent':
				continue
			print('## intent:{}'.format(obj['name']))
			for utterance in obj['utterances']:
				utterance = re.sub(r"\[([^\]]*)\]\(([^\)]*)\)", r"[\2](\1)", utterance)
				print('- {}'.format(utterance))
			print('')

convert_snips_to_rasa()
