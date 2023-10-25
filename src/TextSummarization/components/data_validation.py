import os

class DataValidation:
    def __init__(self, config):
        self.config = config

    def validate_existed_files(self):
        validation_status = None
        all_files = os.listdir(os.path.join('artifacts', 'dataset', 'samsum_dataset'))
        print(self.config['ALL_REQUIRED_FILES'])
        for file in all_files:
            if file not in self.config['ALL_REQUIRED_FILES']:
                validation_status = False
                with open(self.config['STATUS_FILE'], 'w') as f:
                    f.write(f'Vaidation status: {validation_status}')
            else:
                validation_status = True
                with open(self.config['STATUS_FILE'], 'w') as f:
                    f.write(f'Vaidation status: {validation_status}')
        return validation_status
    