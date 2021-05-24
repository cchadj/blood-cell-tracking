import os
import sys
from os.path import basename
from matplotlib import pyplot as plt

from PyQt5 import QtWidgets
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from cell_selector import MplFrameSelector
from shared_variables import csv_file_extensions, video_file_extensions, DATA_FOLDER, OUTPUT_FOLDER

from cell_selector_ui import Ui_MainWindow
from cell_selector_model import CellSelectorModel

# type hinting
from typing import List


# noinspection PyBroadException
class MainWindowUIClass(Ui_MainWindow):
    widgets: List[QtWidgets.QWidget]
    model: CellSelectorModel
    frame_selector: MplFrameSelector

    def __init__(self):
        '''Initialize the super class
        '''
        super().__init__()
        self.model = CellSelectorModel()
        self.frame_selector = None
        self.widgets = []
        self.not_implemented_widgets = []

    def setupUi(self, MW):
        ''' Setup the UI of the super class, and add here code
        that relates to the way we want our UI to operate.
        '''
        super().setupUi(MW)
        self.cell_positions_csv_target_lbl.setText('')
        self.mask_video_file_target_lbl.setText('')
        self.vessel_mask_loaded_target_lbl.setText('')
        self.debugDisplay.append("Please select avi. 'File -> New' or 'Ctr + N'")

        self.widgets.append(self.next_marked_frame_btn)
        self.widgets.append(self.prev_marked_frame_btn)
        self.widgets.append(self.next_frame_btn)
        self.widgets.append(self.prev_frame_btn)
        self.widgets.append(self.load_cell_positions_csv_btn)
        self.widgets.append(self.load_mask_video_btn)
        self.widgets.append(self.load_vessel_mask_btn)
        self.widgets.append(self.cur_frame_txt)

        self.not_implemented_widgets.append(self.load_vessel_mask_btn)
        self.not_implemented_widgets.append(self.load_cell_positions_csv_btn)
        self.not_implemented_widgets.append(self.load_mask_video_btn)

        for widget in self.widgets:
            widget.setDisabled(True)

        for widget in self.not_implemented_widgets:
            widget.setDisabled(True)

    def newVideoSlot(self):
        dialog = QFileDialog()

        data_folder_full_path = os.path.abspath(DATA_FOLDER)
        dialog.setDirectoryUrl(QUrl('file:///' + data_folder_full_path))
        dialog.setFileMode(QFileDialog.ExistingFile)
        file_extensions = 'Videos ('
        for extension in video_file_extensions:
            file_extensions += f'*{extension} '
        file_extensions += ')'
        dialog.setNameFilter(file_extensions)
        dialog.setViewMode(QFileDialog.List)
        if dialog.exec_():
            file_names = dialog.selectedFiles()
        else:
            self.debugDisplay.append(f'No file selected.')
            return

        if len(file_names) == 0:
            self.debugDisplay.append(f'No file selected.')
            return

        try:
            self.model.create_video_session(file_names[0])
            self.frame_selector = MplFrameSelector.fromvideosession(self.model.video_session)
            self.frame_selector.activate()
            # because it's a matplotlib based class we need to call plt show afterwards
            plt.show()
            self.cur_frame_txt.setText(str(self.frame_selector.frame_idx))
            for btn in self.widgets:
                btn.setEnabled(True)

            self.filed_loaded_target_lbl.setText(basename(self.model.video_session.video_oa790_file))
            self.vessel_mask_loaded_target_lbl.setText(basename(self.model.video_session.vessel_mask_confocal_file))
            self.cell_positions_csv_target_lbl.setText(basename(self.model.video_session.cell_position_csv_files[0]))
            self.mask_video_file_target_lbl.setText(basename(self.model.video_session.mask_video_oa790_file))
            self.debugDisplay.append(f"Output csv file will be saved as: '{self.frame_selector.output_file}'")
            self.debugDisplay.append("'Ctr + S' to save")
            self.debugDisplay.append("'File -> Save' as to change output file")
        except:
            print('No such file or file is not readable. Please select other video.')
            self.debugDisplay.append('No such file or file is not readable. Please select other video.')

        print(f'Selected files {file_names}')

    def loadVesselMaskSlot(self):
        self.debugDisplay.append('Functionality not yet implemented')

    def loadCellPositionsCsvSlot(self):
        dialog = QFileDialog()

        data_folder_full_path = os.path.abspath(DATA_FOLDER)
        dialog.setDirectoryUrl(QUrl('file:///' + data_folder_full_path))
        dialog.setFileMode(QFileDialog.ExistingFile)
        file_extensions = 'Csv files ('
        for extension in csv_file_extensions:
            file_extensions += f'*{extension} '
        file_extensions += ')'
        dialog.setNameFilter(file_extensions)
        dialog.setViewMode(QFileDialog.List)
        if dialog.exec_():
            filenames = dialog.selectedFiles()
        else:
            self.debugDisplay.append('No file chosen')
            return

        if len(filenames) == 0:
            self.debugDisplay.append('No file chosen')
            return
        self.model.video_session.append_cell_position_csv_file(filenames[0])
        self.frame_selector.close()

        self.frame_selector = MplFrameSelector.fromvideosession(self.model.video_session)
        self.frame_selector.activate()
        plt.show()

    def loadMaskVidSlot(self):
        self.debugDisplay.append('Functionality not yet implemented')

    def goToNextFrameSlot(self):
        self.frame_selector.frame_idx += 1
        self.cur_frame_txt.setText(str(self.frame_selector.frame_idx))
        print('Next frame')
        plt.show()
        pass

    def goToPrevFrameSlot(self):
        self.frame_selector.frame_idx -= 1
        self.cur_frame_txt.setText(str(self.frame_selector.frame_idx))
        print('Prev frame')
        plt.show()
        pass

    def goToNextMarkedFrameSlot(self):
        self.frame_selector.next_marked_frame()
        self.cur_frame_txt.setText(str(self.frame_selector.frame_idx))
        print('Going to next marked frame slot')
        plt.show()
        pass

    def goToPrevMarkedFrameSlot(self):
        self.frame_selector.prev_marked_frame()
        self.cur_frame_txt.setText(str(self.frame_selector.frame_idx))
        print('Going to prev marked frame slot')
        plt.show()
        pass

    def saveSlot(self):
        self.frame_selector.save()
        print(f'Saved output to {self.frame_selector.output_file}')
        pass

    def saveAsSlot(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setDirectoryUrl(QUrl(os.path.join('file:', os.path.dirname(self.frame_selector.output_file))))
        file_extensions = 'Csv files ('
        for extension in csv_file_extensions:
            file_extensions += f'*{extension} '
        file_extensions += ')'
        dialog.setNameFilter(file_extensions)
        dialog.setViewMode(QFileDialog.List)
        if dialog.exec_():
            filenames = dialog.selectedFiles()
        else:
            self.debugDisplay.append(f'No file selected.')
            return

        if len(filenames) > 0:
            if os.path.exists(filenames[0]):
                # Ask user if it's ok to overwrite existing document
                msg_box = QMessageBox()
                msg_box.setText('File with that name already exists.')
                msg_box.setInformativeText('Are you sure you want to overwrite?')
                msg_box.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
                msg_box.setDefaultButton(QMessageBox.Save)
                ret = msg_box.exec()

                if ret == QMessageBox.Save:
                    is_file_safe_to_save = True
                elif ret == QMessageBox.Cancel:
                    is_file_safe_to_save = False
                    self.debugDisplay.append('File not saved.')
            if is_file_safe_to_save:
                self.frame_selector.output_file = filenames[0]
                self.frame_selector.save()
                self.debugDisplay.append(f'Saved output as: {self.frame_selector.output_file}')
        else:
            self.debugDisplay.append(f'No file selected.')


def main():
    """
    This is the MAIN ENTRY POINT of our application.  The code at the end
    of the mainwindow.py script will not be executed, since this script is now
    our main program.   We have simply copied the code from mainwindow.py here
    since it was automatically generated by '''pyuic5'''.

    """
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MainWindowUIClass()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
