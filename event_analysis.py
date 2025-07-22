# Name: Molecular Dynamics: Desmond Event Analysis...
# Command: pythonrun event_analysis.show_panel
"""
$Revision: 0.0 $

Description:  A parent script for all the Event Analysis panels.
Contributors:  Pat Lorton, Alex Smondyrev, Dmitry Lupyan

Copyright Schrodinger, LLC. All rights reserved.
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from past.utils import old_div
from pathlib import Path

from schrodinger.models.parameters import CompoundParam
from schrodinger.tasks import tasks
import event_analysis_dir.pl_image_tools as pl_tools
import reportlab
import reportlab.lib.colors as rlcolors
# The event_analysis_rc import loads the icons into Qt
from event_analysis_dir import event_analysis_rc  # noqa: F401
from event_analysis_dir import event_analysis_ui
from reportlab import platypus
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

import schrodinger.application.desmond.automatic_analysis_generator as aag
from schrodinger.application.desmond import cms
from schrodinger.application.desmond.event_analysis.tasks import EAFGenerationTask
from schrodinger.application.desmond.event_analysis.tasks import EAFAnalysisTask
from schrodinger.application.desmond.event_analysis.tasks import check_input_file_is_valid
from schrodinger.application.desmond.event_analysis.tasks import make_preprocessor_failure_message
from schrodinger.application.desmond.event_analysis.tasks import EAF_EXT
from schrodinger.application.desmond.event_analysis.tasks import PDF_EXT
from schrodinger.application.desmond.event_analysis.tasks import DEFAULT_PROT_ASL
from schrodinger.application.desmond.event_analysis.tasks import DEFAULT_LIG_ASL
from schrodinger.application.desmond.packages import topo
from schrodinger.application.desmond.packages import traj
from schrodinger.job import jobhandler
from schrodinger.Qt import QtCore
from schrodinger.Qt import QtGui
from schrodinger.Qt import QtWidgets
# If script is running on windows, desmond is not imported.
from schrodinger.structutils import analyze
from schrodinger.structutils import build
from schrodinger.ui.qt import appframework
from schrodinger.ui.qt import config_dialog
from schrodinger.ui.qt import filedialog
from schrodinger.ui.qt.forcefield import forcefield
from schrodinger.ui.qt import swidgets
from schrodinger.ui.qt import table
from schrodinger.ui.qt.decorators import wait_cursor
from schrodinger.ui.sequencealignment.constants import ANNOTATION_RESNUM
from schrodinger.ui.sequencealignment.constants import ANNOTATION_SSBOND
from schrodinger.ui.sequencealignment.constants import COLOR_POSITION
from schrodinger.application.desmond.constants import ORIGINAL_INDEX
from schrodinger.application.desmond.constants import ORIGINAL_MOLECULE_NUMBER
from schrodinger.ui.sequencealignment.sequence_viewer import SequenceViewer
from schrodinger.utils import documentation
from schrodinger.utils import fileutils
from schrodinger.utils import sea
from schrodinger.utils import qapplication
from schrodinger import adapter

SELECTION_STYLESHEET = \
        "QTableView { selection-background-color: rgb(205,222,179);}"

try:
    from schrodinger import maestro
except ImportError:
    maestro = None

styles = reportlab.lib.styles.getSampleStyleSheet()
HeaderStyle = styles["Heading1"]
ParaStyle = styles["Normal"]

# Command modes used to invoke event analysis
GUI = 'gui'
ANALYZE = 'analyze'
REPORT = 'report'


class EANoLigandException(Exception):
    """
    This is a special exception, which is thrown when no ligand was selected
    using 'Auto' option and user have chosen to go back and select ligand using
    ASL option.
    """


#-Functions--------------------------------------------------------------------


class DataThreadLoader(QtCore.QThread):

    def __init__(self, fname):
        QtCore.QThread.__init__(self)
        self.out_data = None
        self.fname = fname

    def run(self):
        self.out_data = load_sea(self.fname)


def load_sea(fname):
    with open(fname, 'r') as fh:
        return sea.Map(fh.read())


class EventAnalysisPanel(QtCore.QObject):
    cms_st = None
    traj_fn = None
    ligand_st = None

    def __init__(self, inp_file=None, test_mode=False):
        if not maestro:
            self.app = qapplication.get_application()
        QtCore.QObject.__init__(self)

        self.protein_asl = aag.PROTEIN_ASL
        self.qw = QtWidgets.QWidget()
        self.ui = event_analysis_ui.Ui_Form()
        self.ui.setupUi(self.qw)
        self.qw.setWindowTitle("Simulation Interactions Diagram")
        self.qw.setWindowIcon(
            QtGui.QIcon(QtGui.QPixmap(':/event_analysis_icon.png')))
        self.panel = None
        self._current_job = None
        self._progress = None
        self.current_jobs = []

        # add icon to load button
        load_icon = QtWidgets.QApplication.style().standardIcon(
            QtWidgets.QStyle.SP_DialogOpenButton)
        self.ui.loadButton.setIcon(load_icon)

        self.ui.loadButton.clicked.connect(self.loadData)
        self.ui.reportButton.clicked.connect(self.export)
        self.ui.helpButton.clicked.connect(self.showHelp)

        self.test_mode = test_mode
        if not test_mode:
            self.qw.show()
        if inp_file:
            self.loadData(inp_file)

    def warning(self, window, title, text):
        if self.test_mode:
            print("WARNING DIALOG: '%s': '%s'" % (title, text))
        else:
            QtWidgets.QMessageBox.warning(window, title, text)

    def info(self, window, title, text):
        if self.test_mode:
            print("INFO DIALOG: '%s': '%s'" % (title, text))
        else:
            QtWidgets.QMessageBox.information(window, title, text)

    @staticmethod
    def str(str_in):
        """
        This is to remove the double quotes in ARK string (ark_str),
        returned a string without enclosed double quotes. str.strip()
        is not used here as it also removes both quotes at the end of
        the following asl example: "res.ptype \"ABC\""
        """
        str_out = str(str_in)
        if str_out.startswith('"') and str_out.endswith('"'):
            return str_out[1:-1]
        return str_out

    def sanitizeASLstring(self, asl):
        """ ASL selections should not contain double quotes """
        asl_str = self.str(asl)
        return asl_str.replace('\"', '\'').strip()

    def showHelp(help):
        """ Call Help topic"""
        documentation.show_topic("DESMOND_SIMULATION_INTERACTIONS_DIAGRAM")

    def loadData(self, filename=None):
        """This loads a pre-computed analysis file."""

        if not filename:
            filename = filedialog.get_open_file_name(
                parent=self.qw,
                caption="Select Event Analysis or Trajectory File",
                id='event_analysis',
                filter="Any Valid File (*.eaf *-out.cms);;"
                "Event Analysis File (*.eaf);;"
                "Trajectory File (*-out.cms)")

        if not filename:
            return

        if (os.path.splitext(filename)[1] == ".eaf"):
            self.loadEAF(filename)
            self.ui.reportButton.setEnabled(True)
        else:
            self.loadCalculation(filename)

    def getJobnameTypeLigandASL(self, lc_ui, cms_st):
        """
        Based on the dialog settings, return the jobtype, the ligand_asl,
        and the list of keywords that go to the backend for that job type.

        :param lc_ui:  UI file for the launch information
        :type  lc_ui:  ui module which wraps widgets

        :param cms_st: A CMS to find a ligand within
        :type  cms_st: cms.Cms object

        :return: jobtype, ligand ASL, list of keywords
        :rtype:  str, str, [sea.Map(), ...]
        """
        if lc_ui.selectAnalysisCombo.currentIndex() == 0:
            l, ligand_asl = self.getPLKeywords(lc_ui.pl_is_ui, cms_st)
            if len(l) == 0:
                return None, None, None
            return "pl", ligand_asl, l


#TODO:  Once other panels are operational, uncomment these
#        if lc_ui.selectAnalysisCombo.currentIndex() == 1:
#            l = self.getHBEKWList(lc_ui.hb_e_ui, cms_st)
#            if not l:
#                return None, None, None
#            return "hbond", None, l
#        if lc_ui.selectAnalysisCombo.currentIndex() == 2:
#            l = self.getSBEKWList(lc_ui.sb_e_ui, cms_st)
#            if not l:
#                return None, None, None
#            return "ionic", None, l
#        if lc_ui.selectAnalysisCombo.currentIndex() == 3:
#            l = self.getHPEKWList(lc_ui.hp_e_ui, cms_st)
#            if not l:
#                return None, None, None
#            return "hydrophobic", None, l

    def getPLKeywords(self, lc_ui, cms_st):
        """
        Get the list of all keywords based on what's in the lc_ui for a
        PL job.

        :param lc_ui:  UI file for the launch information
        :type  lc_ui:  ui module which wraps widgets

        :param cms_st: A CMS to find a ligand within
        :type  cms_st: cms.Cms object

        :return: list of keywords, ligand ASL
        :rtype:  [sea.Map(), ...], str
        """

        ligand_st, ligand_asl = self.getLigandInfoPLKW(cms_st, lc_ui)

        want_protein = (lc_ui.proteinSelect.currentText() != "None")
        protein_asl = self.protein_asl
        if not want_protein:
            protein_asl = None

        frame, ref_struct = 0, None
        if lc_ui.comboBox.currentIndex() == 0:
            frame = lc_ui.spinBox.value()
        else:
            ref_struct = str(lc_ui.lineEdit.text())

        want_rmsd = bool(lc_ui.rmsdCB.isChecked() and protein_asl)
        want_prmsf = bool(lc_ui.prmsfCB.isChecked() and protein_asl)
        want_lrmsf = bool(lc_ui.lrmsfCB.isChecked() and ligand_asl)
        want_pli = bool(lc_ui.pliCB.isChecked() and ligand_asl and protein_asl)
        want_ltorsion = bool(lc_ui.ltCB.isChecked() and ligand_asl)
        want_lprops = bool(lc_ui.lpropsCB.isChecked() and ligand_asl)
        want_ppi = bool(protein_asl)

        return aag.getPLISKWList(cms_st,
                                 ligand_st,
                                 ligand_asl,
                                 ref_struct,
                                 frame,
                                 protein_asl=protein_asl,
                                 want_rmsd=want_rmsd,
                                 want_prmsf=want_prmsf,
                                 want_lrmsf=want_lrmsf,
                                 want_pli=want_pli,
                                 want_ppi=want_ppi,
                                 want_ltorsion=want_ltorsion,
                                 want_lprops=want_lprops), ligand_asl

    def setupCalcWindow(self):
        import event_analysis_dir.new_calculations_ui as new_calculations_ui

        launchCalcW = QtWidgets.QDialog(self.qw)
        launchCalcW.setWindowModality(QtCore.Qt.WindowModal)
        lc_ui = new_calculations_ui.Ui_Dialog()
        lc_ui.setupUi(launchCalcW)
        lc_ui.buttonBox.addButton("Run", QtWidgets.QDialogButtonBox.AcceptRole)
        layout = QtWidgets.QVBoxLayout(lc_ui.frame)
        lc_ui.pl_is_ui = self.addPLIWidget(layout)
        #lc_ui.hb_e_ui  = self.addHBondWidget(layout)
        #lc_ui.sb_e_ui  = self.addSBridgeWidget(layout)
        #lc_ui.hp_e_ui  = self.addHydrophobicWidget(layout)
        # hide top frame since we only support one PL Interact
        # Survey calculation type for now (DESMOND-3125)
        lc_ui.frame_2.setHidden(True)
        lc_ui.line.setHidden(True)
        layout.addStretch()

        return launchCalcW, lc_ui

    def _setReferenceFile(self, w):

        def f():
            launchCalcW = QtWidgets.QDialog(self.qw)
            filenameTxt = filedialog.get_open_file_name(
                parent=launchCalcW,
                id="sea",
                caption="Select Reference Maestro File",
                filter="Maestro File(*.mae*)")

            if not filenameTxt:
                return
            else:
                w.lineEdit.setText(filenameTxt)

        return f

    def addPLIWidget(self, layout):
        import event_analysis_dir.pl_is_new_calculations_ui as pl_is_new_calculations_ui
        pl_is_w = QtWidgets.QWidget()
        pl_is_ui = pl_is_new_calculations_ui.Ui_NewCalculations()
        pl_is_ui.setupUi(pl_is_w)
        layout.addWidget(pl_is_w)
        self.options_widget = pl_is_w

        def setChecked(w):

            def f():
                if w.isEnabled():
                    w.setChecked(True)

            return f

        for widget in [
                pl_is_ui.rmsdCB, pl_is_ui.prmsfCB, pl_is_ui.lrmsfCB,
                pl_is_ui.pliCB, pl_is_ui.ltCB, pl_is_ui.lpropsCB
        ]:
            pl_is_ui.selectAllB.clicked.connect(setChecked(widget))
        pl_is_ui.browseB.clicked.connect(self._setReferenceFile(pl_is_ui))
        pl_is_ui.proteinSelect.currentIndexChanged.connect(
            self.genSelectASLFunc(pl_is_ui.proteinSelect,
                                  "Protein",
                                  self.protein_asl,
                                  panel=pl_is_ui))
        pl_is_ui.ligandSelect.currentIndexChanged.connect(
            self.genSelectASLFunc(pl_is_ui.ligandSelect,
                                  "Ligand",
                                  "",
                                  panel=pl_is_ui))

        return pl_is_ui

    def addHBondWidget(self, layout):
        import event_analysis_dir.hb_explorer_new_calculations_ui as hb_explorer_new_calculations_ui
        hb_e_w = QtWidgets.QWidget()
        hb_e_ui = hb_explorer_new_calculations_ui.Ui_NewCalculations()
        hb_e_ui.setupUi(hb_e_w)
        layout.addWidget(hb_e_w)
        hb_e_w.setVisible(False)
        return hb_e_ui

    def addSBridgeWidget(self, layout):
        import event_analysis_dir.sb_explorer_new_calculations_ui as sb_explorer_new_calculations_ui
        sb_e_w = QtWidgets.QWidget()
        sb_e_ui = sb_explorer_new_calculations_ui.Ui_NewCalculations()
        sb_e_ui.setupUi(sb_e_w)
        layout.addWidget(sb_e_w)
        sb_e_w.setVisible(False)
        return sb_e_ui

    def addHydrophobicWidget(self, layout):
        import event_analysis_dir.hp_explorer_new_calculations_ui as hp_explorer_new_calculations_ui
        hp_e_w = QtWidgets.QWidget()
        hp_e_ui = hp_explorer_new_calculations_ui.Ui_NewCalculations()
        hp_e_ui.setupUi(hp_e_w)
        layout.addWidget(hp_e_w)
        hp_e_w.setVisible(False)
        return hp_e_ui

    def genSelectASLFunc(self, widget, txt, default, panel=None):

        def selectASLFunc(index):
            if txt == 'Ligand' and index == 2:
                self.disableLigandOptions(panel)
            elif txt == 'Ligand':
                self.enableLigandOptions(panel)
            if txt == 'Protein' and index == 2:
                self.disableProteinOptions(panel)
            elif txt == 'Protein':
                self.enableProteinOptions(panel)
            if index == 1:  #If they hit select ASL
                asl, ok = None, True
                if maestro:
                    asl = maestro.atom_selection_dialog("", current_asl=default)
                else:
                    launchCalcW = QtWidgets.QDialog(self.qw)
                    asl, ok = QtWidgets.QInputDialog.getText(
                        launchCalcW, "%s ASL" % txt,
                        "Please enter a valid ASL for the %s:" % txt,
                        QtWidgets.QLineEdit.Normal, default)
                    asl = self.sanitizeASLstring(asl)
                    # if protein selection changes, update the protein_asl
                print(txt, index)
                if txt == 'Protein' and index == 2:
                    self.protein_asl = None
                elif txt == 'Protein':
                    self.protein_asl = '(' + self.sanitizeASLstring(asl) + ')'
                if not asl or not ok:
                    widget.setCurrentIndex(0)
                    return
                widget.addItem(asl)
                widget.setCurrentIndex(widget.count() - 1)

        return selectASLFunc

    def disableLigandOptions(self, panel):
        """
        This function is used to uncheck and disable options that can not
        be used when 'Ligand' selection is 'None'.

        :param panel: instance of new calculation panel
        :type panel: `NewCalculations`
        """

        for label, widget in [(panel.label_6, panel.lrmsfCB),
                              (panel.label_7, panel.pliCB),
                              (panel.label_9, panel.ltCB),
                              (panel.label_12, panel.lpropsCB)]:
            label.setEnabled(False)
            widget.setChecked(False)
            widget.setEnabled(False)

    def disableProteinOptions(self, panel):
        """
        This function is used to uncheck and disable options that can not
        be used when 'Protein' selection is 'None'.

        :param panel: instance of new calculation panel
        :type panel: `NewCalculations`
        """
        for label, widget in [(panel.label_4, panel.rmsdCB),
                              (panel.label_5, panel.prmsfCB),
                              (panel.label_7, panel.pliCB)]:
            label.setEnabled(False)
            widget.setChecked(False)
            widget.setEnabled(False)

    def enableLigandOptions(self, panel):
        """
        This function is used to enable options that were previously
        disabled when 'Ligand' selection was 'None'.

        :param panel: instance of new calculation panel
        :type panel: `NewCalculations`
        """

        for label, widget in [(panel.label_6, panel.lrmsfCB),
                              (panel.label_7, panel.pliCB),
                              (panel.label_9, panel.ltCB),
                              (panel.label_12, panel.lpropsCB)]:
            label.setEnabled(True)
            widget.setEnabled(True)

    def enableProteinOptions(self, panel):
        """
        This function is used to enable options that were previously
        disabled when 'Protein' selection was 'None'.

        :param panel: instance of new calculation panel
        :type panel: `NewCalculations`
        """
        for label, widget in [(panel.label_4, panel.rmsdCB),
                              (panel.label_5, panel.prmsfCB),
                              (panel.label_7, panel.pliCB)]:
            label.setEnabled(True)
            widget.setEnabled(True)

    def changeTab(self, index):
        if index == self.old_tab_index:
            return
        self.old_tab_index = index
        self.options_widget.hide()
        # This method must be broken. See PANEL-13352
        if index == 0:
            self.options_widget = pl_is_w  # noqa F821
            pl_is_w.show()  # noqa F821

        elif index == 1:
            self.options_widget = hb_e_w  # noqa F821
            hb_e_w.show()  # noqa F821

        elif index == 2:
            self.options_widget = sb_e_w  # noqa F821
            sb_e_w.show()  # noqa F812

        elif index == 3:
            self.options_widget = hp_e_w  # noqa F821
            hp_e_w.show()  # noqa F821

    def writeTempEAF(self, traj_filename, ligand_asl, kws):
        schrod_tmp = fileutils.get_directory_path(fileutils.TEMP)
        eaf_tmp_fname = fileutils.get_next_filename(
            os.path.join(schrod_tmp, "EAF_tmp.eaf"), "")
        eaf = open(eaf_tmp_fname, 'w')
        m = sea.Map()
        m["Trajectory"] = traj_filename
        m["Keywords"] = kws
        if ligand_asl:
            m["LigandASL"] = ligand_asl
        if self.protein_asl:
            m["ProteinASL"] = self.protein_asl
        eaf.write(str(m))
        eaf.close()
        return eaf_tmp_fname

    def getEAFFilename(self, jobname, jobname_type):
        eaf_fname = os.path.split(
            fileutils.get_next_filename(
                os.path.join(os.getcwd(),
                             "%s_%s_.eaf" % (jobname, jobname_type)), ""))[1]
        if eaf_fname in self.current_jobs:
            eaf_fname = fileutils.get_next_filename(eaf_fname, "")
        return eaf_fname

    def getJobInfo(self, traj_filename):
        jobname = os.path.splitext(os.path.split(traj_filename)[1])[0]
        sg = config_dialog.StartDialog(self.qw,
                                       jobname=jobname,
                                       default_disp=appframework.DISP_APPEND)
        sd_params = sg.activate()
        if not sd_params:
            return None
        host = getattr(sd_params, "host")
        jobname = getattr(sd_params, "jobname")
        disp = getattr(sd_params, "disp")
        return host, jobname, disp

    def progressDialog(self):
        """Shows the progress dialog.

        We make the progress dialog modal so that it blocks the SID, but call
        show() on it, so that it does not block the rest of Maestro.
        """
        progress = QtWidgets.QProgressDialog("Waiting For Job to Finish",
                                             "Hide", 0, 0, self.qw)
        progress.setWindowTitle("Please Wait")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.show()
        return progress

    def validProtein(self, cms_st, lc_ui):
        """
        Check to if protein exists.
        """
        if (lc_ui.pl_is_ui.proteinSelect.currentText() == "None"):
            self.protein_asl = None
            return True
        atom_list = analyze.evaluate_asl(cms_st, self.protein_asl)
        if (lc_ui.pl_is_ui.proteinSelect.currentText()
                == "Auto") and not len(atom_list):
            self.protein_asl = None
            return True
        if not len(atom_list):
            self.warning(
                self.qw, "Invalid Protein",
                "Protein with ASL: %s cannot be detected. " %
                (self.protein_asl) + "  Please adjust your definition.")
            return False
        else:
            return True

    def loadCalculation(self, traj_filename):
        """
        This function is used to create a .eaf file containing all
        simulation analysis information for a given panel.  If it's run
        multiple times on the same simulation, it will store the information
        in the same .eaf file, and prompt them to pick a panel when they
        select that .eaf file.
        """

        self.ui.reportButton.setEnabled(False)
        try:
            _, cms_st = topo.read_cms(traj_filename)
        except:
            self.warning(
                self.qw, "Invalid Trajectory",
                "The trajectory failed to load:\n %s" % (traj_filename))
            return

        self._importTrajectoryIntoMaestro(traj_filename)

        launchCalcW, lc_ui = self.setupCalcWindow()

        self.old_tab_index = 0
        lc_ui.selectAnalysisCombo.currentIndexChanged.connect(self.changeTab)

        if not launchCalcW.exec():
            return

        if not self.validProtein(cms_st, lc_ui):
            return

        self.ui.tabWidget.clear()

        for a in cms_st.atom:
            a.property[ORIGINAL_INDEX] = a.index
            a.property[ORIGINAL_MOLECULE_NUMBER] = a.molecule_number

        try:
            jobname_type, ligand_asl, kws = \
                self.getJobnameTypeLigandASL(lc_ui, cms_st)
        except EANoLigandException:
            return self.loadCalculation(traj_filename)

        job_args = self.getJobInfo(traj_filename)
        if job_args:
            host, jobname, disp = job_args
        else:
            return

        # assuming the trajectory and cms file are in the same folder
        traj_path = topo.find_traj_path_from_cms_path(traj_filename)
        if traj_path is None:
            self.warning(
                self.qw, 'Invalid CMS file',
                'Could not locate a trajectory directory for given CMS file: {}.'
                .format(traj_filename))
            return
        eaf_tmp_fname = self.writeTempEAF(traj_filename, ligand_asl, kws)
        eaf_fname = self.getEAFFilename(jobname, jobname_type)

        schrodinger_run = os.path.join(os.environ.get("SCHRODINGER"), "run")
        cmd = [
            schrodinger_run,
            "analyze_simulation.py",
            traj_filename,
            traj_path,
            eaf_fname,
            eaf_tmp_fname,
            '-HOST',
            host,
            '-JOBNAME',
            Path(eaf_fname).stem,
        ]
        if forcefield.get_use_custom_forcefield_preference():
            opls_dir = forcefield.get_custom_opls_dir()
            rc = forcefield.validate_opls_dir(opls_dir, parent=self.qw)
            if rc == forcefield.OPLSDirResult.ABORT:
                return
            elif rc == forcefield.OPLSDirResult.VALID:
                cmd += ['-OPLSDIR', opls_dir]
        if maestro:
            # make the job visible by the user via the Job Monitor panel:
            cmd += ['-PROJ', maestro.project_table_get().project_name]

        # Start the job
        try:
            job = jobhandler.launch_job_with_callback(cmd,
                                                      self._onJobCompleted,
                                                      show_failure_dialog=False)
        except RuntimeError as err:
            self.warning(self.qw, "Launch failure",
                         "Failed to start the job:\n\n%s" % err)
            return
        # Store current job so only the most recently launched job is loaded
        # into the panel
        self._current_job = job
        self.current_jobs.append(eaf_fname)
        self._progress = self.progressDialog()

    def setupLigandTable(self, ligand_list):
        model = table.StructureDataViewerModel(rows=1, columns=len(ligand_list))
        column = 0
        for column, lig in enumerate(ligand_list):
            model.setCellValue(0, column, lig.st)

        ligtable = table.DataViewerTable(model,
                                         fill='none',
                                         cell_width=250,
                                         cell_height=250)
        ligtable.setStyleSheet(SELECTION_STYLESHEET)

        ligtable.setAspectRatioPolicy(True, resize=False)
        ligtable.setGangPolicy('both')
        ligtable.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        delegate = customDelegate(ligtable)  # noqa: F841
        return ligtable

    def pickLigandDialog(self, ligtable, ligand_list):
        qd = QtWidgets.QDialog()
        qd.resize(50 + len(ligand_list) * 250, 350)
        layout = QtWidgets.QVBoxLayout(qd)
        layout.addWidget(ligtable)
        qd.setWindowTitle("Please select the Ligand to use")
        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | \
                QtWidgets.QDialogButtonBox.Cancel)

        def try_accept():
            try:
                ligtable.selectionModel().selection().indexes()[0]
                qd.accept()
            except:  #No selection
                self.warning(
                    qd, "No Scaffold Selected",
                    "You must select at least one scaffold to launch RGA.")
                return False

        buttonBox.accepted.connect(try_accept)
        buttonBox.rejected.connect(qd.reject)
        layout.addWidget(buttonBox)
        return bool(qd.exec())

    def pickLigandFromList(self, ligand_list):
        ligtable = self.setupLigandTable(ligand_list)
        if not self.pickLigandDialog(ligtable, ligand_list):
            return None, None

        select_idx = ligtable.selectionModel().selection().indexes()[0].column()
        ligand_st = ligand_list[select_idx].st
        ligand_asl = str(ligand_list[select_idx].ligand_asl)
        return ligand_st, ligand_asl

    def getLigandAutoSelect(self, cms_st):
        ligand_list = analyze.find_ligands(cms_st)
        # only filter ligand list when it has more than 1 ligand
        if len(ligand_list) > 1:
            ligand_list = aag.removeAminoAcids(ligand_list)
        else:
            ligand_list = ligand_list[:]
        if len(ligand_list) == 0:
            reply = QtWidgets.QMessageBox.question(
                self.qw, "No Ligands Found",
                "We couldn't detect a ligand in your input system, "
                "you can manually define it via ASL. Would you like "
                "to continue with only protein-related analyses?",
                QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                return None, None
            else:
                raise EANoLigandException

        elif len(ligand_list) > 1:
            ligand_st, ligand_asl = self.pickLigandFromList(ligand_list)
            if not ligand_st:
                return None, None
        else:
            ligand_st = ligand_list[0].st
            ligand_asl = str(ligand_list[0].ligand_asl)
        return ligand_st, ligand_asl

    def getLigandInfoPLKW(self, cms_st, ui):
        ligand_st, ligand_asl = None, None
        if str(ui.ligandSelect.currentText()) == "Auto":
            ligand_st, ligand_asl = self.getLigandAutoSelect(cms_st)
            if not ligand_st:
                return None, None

        elif str(ui.ligandSelect.currentText()) != "None":
            #If it's not "Auto" or "None", then it's a valid ASL
            atom_list = analyze.evaluate_asl(cms_st,
                                             str(ui.ligandSelect.currentText()))
            ligand_st = cms_st.extract(atom_list)
            ligand_asl = str(ui.ligandSelect.currentText())

        return ligand_st, ligand_asl

    def getHBEKWList(self, ui, cms_st):
        """
        This function generates a keyword list to be run for calculations
        when the user has selected a H-Bond Explorer calculation type.
        """
        lst = sea.List()
        ark_kw = sea.Map()
        ark_kw["HBonds"] = sea.Map()
        ark_kw["HBonds"]["ASL1"] = "(all)"
        ark_kw["HBonds"]["ASL2"] = "(all)"
        ark_kw["HBonds"]["Panel"] = 'hbond_explorer'
        ark_kw["HBonds"]["Tab"] = 'hbond_explorer'
        ark_kw["HBonds"]["ReturnHBonds"] = "True"
        ark_kw["HBonds"]["Type"] = "ASL"
        ark_kw["HBonds"]["Unit"] = "H-Bonds"
        lst.append(ark_kw)

        return lst

    def getSBEKWList(self, ui, cms_st):
        """
        This function generates a keyword list to be run for calculations
        when the user has selected a Salt Bridge Explorer calculation type.
        """

        lst = sea.List()
        ark_kw = sea.Map()
        ark_kw["SaltBridges"] = sea.Map()
        ark_kw["SaltBridges"]["ASL1"] = "(protein)"
        ark_kw["SaltBridges"]["ASL2"] = ""
        ark_kw["SaltBridges"]["Panel"] = 'sbridge_explorer'
        ark_kw["SaltBridges"]["Tab"] = 'sbridge_explorer'
        ark_kw["SaltBridges"]["Unit"] = "Salt Bridges"
        ark_kw["SaltBridges"]["Cutoff"] = 3.2
        lst.append(ark_kw)

        return lst

    def getHPEKWList(self, ui, cms_st):
        """
        This function generates a keyword list to be run for calculations
        when the user has selected a HPhobic Explorer calculation type.
        """

        lst = sea.List()
        ark_kw = sea.Map()
        ark_kw["HydrophobicInteract"] = sea.Map()
        ark_kw["HydrophobicInteract"]["ASL1"] = "(protein)"
        ark_kw["HydrophobicInteract"]["ASL2"] = ""
        ark_kw["HydrophobicInteract"]["Panel"] = 'hphobe_explorer'
        ark_kw["HydrophobicInteract"]["Tab"] = 'hphobe_explorer'
        ark_kw["HydrophobicInteract"]["Unit"] = "Hydrophobic Interactions"
        lst.append(ark_kw)

        return lst

    def _onJobCompleted(self, job):
        if not self._current_job or self._current_job.JobId != job.JobId:
            return
        if self._progress:
            self._progress.close()
        if job.succeeded():
            out_file = job.Name + ".eaf"
            self.current_jobs.remove(out_file)
            self.loadEAF(out_file)
            self.ui.reportButton.setEnabled(True)
        else:
            log_file = job.Name + ".log"
            self.warning(self.qw, "Job Failed",
                    "The job failed, please inspect the log file:\n%s" %\
                            log_file)
        self._current_job = None

    def getTrajectoryFromEAF(self, cfg, filename):
        trajectory = self.str(cfg["Trajectory"])
        #Check if literal path is good
        if not os.path.isfile(trajectory):
            #Check to see if the trajectory is in the current directory
            trajectory_nopath = os.path.split(trajectory)[1]
            trajectory_dir = os.path.split(os.path.split(trajectory)[0])[1]
            eaf_path = os.path.split(filename)[0]
            trajectory = os.path.join(eaf_path, trajectory_nopath)
            if not os.path.isfile(trajectory):
                #check to see if the trajectory *directory* is in the current directory
                trajectory = os.path.join(eaf_path, trajectory_dir,
                                          trajectory_nopath)
                if not os.path.isfile(trajectory):
                    #Let the user select the trajectory
                    if not self.test_mode:
                        filename = filedialog.get_open_file_name(
                            parent=self.qw,
                            id='event_analysis',
                            caption="Select Trajectory File",
                            filter="Trajectory File (*-out.cms)")
                    elif self.test_mode:
                        filename = None
                    if not filename:
                        self.warning(
                            self.qw, "Valid Trajectory Required",
                            "The provided trajectory file is invalid.")
                        return None
                    trajectory = filename
        return trajectory

    def getDesmondTrajectory(self, cms_st, trajectory):
        idx_filename = cms_st.property['s_chorus_trajectory_file']
        if os.path.split(idx_filename)[0] == '':
            idx_filename = os.path.join(
                os.path.split(trajectory)[0], idx_filename)
        try:
            idx_fhandle = open(idx_filename)
        except:
            self.warning(
                self.qw, "Invalid CMS File",
                "The corresponding .idx file \"%s\" cannot be found" %
                idx_filename)
            return
        trj_dir = None
        trajectory_dir_RE = re.compile(r'^trajectory\s*=\s*(.*)', re.I)
        for line in idx_fhandle:
            m = trajectory_dir_RE.search(line)
            if m:
                trj_dir = m.group(1)
                break
        if os.path.split(trj_dir)[0] == '':
            trj_dir = os.path.join(os.path.split(trajectory)[0], trj_dir)

        return traj.read_traj(trj_dir)

    def kwToTabDict(self, cfg):
        # It might be better to have a wrapper object with
        # properties/attributes and then have the panel decide what tabs to
        # open accordingly
        tab_dict = defaultdict(list)
        panel_set = set()
        for kw in cfg["Keywords"]:

            # kw only has one item, but it can't be indexed into, so we have to
            # iterate here
            t = None
            for skw in kw:
                t = skw

            special_result_type = {"ProtLigInter", "HydrophobicInteract"}
            if "Result" not in kw[t] and t not in special_result_type:
                continue
            panel_set.add(self.str(kw[t]["Panel"]))

            tab = self.str(kw[t]["Tab"])
            tab_dict[tab].append(kw)

            # special case
            # read Ligand Hbond data into pl_inter_tab dict dictionary so
            # that we can display intramolecular hbonds on LID
            if t == 'LigandHBonds' and 'pl_inter_tab' in tab_dict:
                pl_data = tab_dict['pl_inter_tab'][0]
                pl_data['ProtLigInter']['LigandHBonds'] = \
                                              kw['LigandHBonds']['Result']
            # special case
            # in order to be able to plot Ligand RMSD in L-Properties
            # panel, we need to extract it from 'RMSD' kewords and put
            # bundle it with l-prep 'panel' type
            if t == 'RMSD' and kw['RMSD']['Type'].val == "Ligand" and \
                             'FitBy' not in list(kw['RMSD']):
                tab_dict['l_properties_tab'].append(kw)

        # if L-Props were not initially selected, we don't need l_props
        # in the tab_dict. so only if one one record with 'l_properties_tab'
        # exits, then remove it all together
        if len(tab_dict.get('l_properties_tab', [])) == 1:
            del tab_dict['l_properties_tab']

        panel = None
        if len(panel_set) < 1:
            self.warning(
                self.qw, "No results present",
                "This file doesn't contain any Event Analysis results.")
            return None, panel
        elif len(panel_set) > 1:
            # TODO:  We will allow them to select which panel they want to load
            self.warning(
                self.qw, "Multiple results present",
                "This file contains results for multiple analysis panels.")
            return None, panel
        else:
            panel = panel_set.pop()

        return tab_dict, panel

    def setupPLISPanel(self, tab_dict, cms_st, cms_tor):
        import event_analysis_dir.pl_interact_survey as pl_interact_survey

        self.interactSurveyPanel = pl_interact_survey.PLInteractSurvey()
        self.interactSurveyPanel.cms_ct = cms_st

        for tab_name, func in [
            ('pl_rmsd_tab', self.interactSurveyPanel.pl_rmsd_tab),
            ('p_sse_tab', self.interactSurveyPanel.p_sse_tab),
            ('p_rmsf_tab', self.interactSurveyPanel.p_rmsf_tab),
            ('l_rmsf_tab', self.interactSurveyPanel.l_rmsf_tab),
            ('pl_inter_tab', self.interactSurveyPanel.pl_cont_tab),
            ('pl_inter_tab', self.interactSurveyPanel.lp_cont_tab),
            ('l_torsions_tab', self.interactSurveyPanel.l_torsions_tab),
            ('l_properties_tab', self.interactSurveyPanel.l_props_tab)
        ]:
            if tab_name in tab_dict:
                if tab_name == 'l_torsions_tab':
                    func(self, tab_dict[tab_name], cms_tor)
                else:
                    func(self, tab_dict[tab_name])
        # This call needs to be made after tabs are created in order to show
        # 'contacts' on p-rmsf plot.
        if 'p_rmsf_tab' in tab_dict:
            self.interactSurveyPanel.update_p_rmsf_plot(0)

    def loadEAF(self, filename, cms_class=cms.Cms):

        base_name = os.path.basename(filename)
        self.ui.tabWidget.clear()
        self.ui.reportButton.setEnabled(True)
        self.ui.filenameLabel.setText(base_name)
        self.ui.filenameLabel.setToolTip(os.path.abspath(filename))
        win_title = "Simulation Interactions Diagram (%s)" % base_name.strip(
            '.eaf')
        self.qw.setWindowTitle(win_title)

        progress = QtWidgets.QProgressDialog("Loading in data...", "", 0, 0,
                                             self.qw)
        progress.setWindowTitle("Please Wait")

        thread = DataThreadLoader(filename)
        thread.finished.connect(progress.close)
        if self.test_mode:
            thread.run()
        else:
            thread.start()
            progress.exec()

        with wait_cursor:
            cfg = thread.out_data
            if cfg is None:
                return
            if "Keywords" not in cfg:
                self.warning(
                    self.qw, "No keywords present",
                    "The file you've chosen to import contains no keywords")
                return

            trajectory = self.getTrajectoryFromEAF(cfg, filename)
            if not trajectory:
                return
            self._importTrajectoryIntoMaestro(trajectory)
            _, cms_st = topo.read_cms(trajectory)
            self.traj_fn = trajectory
            self.eaf_fn = filename
            for a in cms_st.atom:
                a.property[ORIGINAL_INDEX] = a.index
                a.property[ORIGINAL_MOLECULE_NUMBER] = a.molecule_number
            self.cms_st = cms_st
            if "LigandASL" in cfg:
                self.ligand_asl = self.sanitizeASLstring(cfg["LigandASL"].val)
            else:
                self.ligand_asl = None

            if "ProteinASL" in cfg:
                self.protein_asl = self.sanitizeASLstring(cfg["ProteinASL"])
            else:
                self.protein_asl = '(protein)'
            cms_tor = cms_class(trajectory)
            if "TrajectoryInterval_ps" in cfg:
                cms_st.frame_time = old_div(
                    float(self.str(cfg["TrajectoryInterval_ps"])), 1000.0)
            else:
                tr = self.getDesmondTrajectory(cms_st, trajectory)
                cms_st.frame_time = old_div((tr[1].time - tr[0].time), 1000.0)

            if "TrajectoryNumFrames" in cfg:
                self.total_sim_time = cms_st.frame_time * \
                                      float(self.str(cfg["TrajectoryNumFrames"]))

            tab_dict, panel = self.kwToTabDict(cfg)
            if not panel:
                return
            self.panel = panel

            if self.panel == 'pl_interact_survey':
                self.setupPLISPanel(tab_dict, cms_st, cms_tor)

    def export(self):
        # This function is used to create pdf report, export plots to image
        # files and export data in user readable format.

        import event_analysis_dir.export_options_ui as export_options_ui

        exportOptions = QtWidgets.QDialog(self.qw)
        exportOptions.setWindowModality(QtCore.Qt.WindowModal)
        ex_opt_ui = export_options_ui.Ui_Dialog()
        ex_opt_ui.setupUi(exportOptions)

        if not exportOptions.exec():
            return

        doReport = ex_opt_ui.reportCB.isChecked()
        doPlots = ex_opt_ui.plotsCB.isChecked()
        doData = ex_opt_ui.dataCB.isChecked()

        export_dir = fileutils.get_next_filename(
            os.path.join(os.getcwd(), "data"), "")
        if not os.path.isfile(export_dir):
            os.mkdir(export_dir)

        if doReport:
            self.generateReport(export_dir)
        if doPlots:
            plot_dir = os.path.join(export_dir, "images")
            if not os.path.isfile(plot_dir):
                os.mkdir(plot_dir)
            if self.panel == 'pl_interact_survey':
                self.interactSurveyPanel.export_plots(plot_dir)
            elif self.panel == 'hbond_explorer':
                pass
            elif self.panel == 'sbridge_explorer':
                pass
            elif self.panel == 'hphobe_explorer':
                pass
        if doData:
            data_dir = os.path.join(export_dir, "raw-data")
            if not os.path.isfile(data_dir):
                os.mkdir(data_dir)
            if self.panel == 'pl_interact_survey':
                self.interactSurveyPanel.export_data(data_dir)
            elif self.panel == 'hbond_explorer':
                pass
            elif self.panel == 'sbridge_explorer':
                pass
            elif self.panel == 'hphobe_explorer':
                pass
        self.info(self.qw, "Report Written", "Report dir: '%s'." % export_dir)

    def reportAddTopImageAndText(self, Elements):
        logo_path = os.path.join(os.environ.get("MMSHARE_EXEC"), "..", "..",
                                 "python", "scripts", "event_analysis_dir",
                                 "schrodinger_logo.png")
        self.logo_img = platypus.Image(logo_path, 1.53 * inch, 0.45 * inch)
        self.logo_img.hAlign = 'RIGHT'
        Elements.append(self.logo_img)
        print_disclaimer = '<br/>* The configuration file (-out.cfg) ' +\
                           'was not found. Keep it in same directory as .aef ' +\
                           'file.'
        self.header(Elements, "Simulation Interactions Diagram Report")
        self.p(Elements, "<u>Simulation Details</u>")
        Elements.append(platypus.Spacer(1, 12))
        path, filename = os.path.split(self.traj_fn)
        job_name = filename[:-8]
        job_type = 'Unknown*'
        ncpus = 'Unknown*'
        temperature = 300.
        ensemble = 'Unknown*'
        total_atoms = self.cms_st.atom_total
        total_charge = self.cms_st.formal_charge
        # get water molecules
        atom_list = analyze.evaluate_asl(self.cms_st, 'water')
        self.water_num = 0
        if atom_list:
            water_st = self.cms_st.extract(atom_list)
            self.water_num = water_st.mol_total

        if 's_m_title' in self.cms_st.property:
            entry_title = \
                 self.cms_st.property['s_m_title'].replace('(full system)', '').strip()
        else:
            entry_title = '<i>Unknown</i>'
        try:
            simulation_time = '%0.3f' % self.total_sim_time
        except:
            simulation_time = '%0.3f' % 0.0
        # may be put these into a table later
        cfg_file = self.traj_fn[:-3] + "cfg"
        if not os.path.isfile(cfg_file):
            #try relative path if absolute doesn't exist
            cfg_file = filename[:-3] + 'cfg'

        if cfg_file.find('replica') != -1:
            job_type = 'FEP'
        elif os.path.isfile(cfg_file):
            print_disclaimer = ''
            cfg = sea.Map(open(cfg_file, "r").read())
            job_name = os.path.split(cfg_file[:-8])[1]
            #ensamble type (NPT, NVT, etc)
            try:
                job_type = str(cfg.app.val)
                ensemble = str(cfg.ORIG_CFG.ensemble.class_.val)
            except:
                try:
                    job_type = str(cfg.app.val)
                    ensemble = str(cfg.ORIG_CFG.ensemble.val)
                except:
                    pass
            # get number of cpus
            cpus = cfg.ORIG_CFG.cpu.val
            if isinstance(cpus, int):
                ncpus = cpus
            else:
                ncpus = cpus[0] * cpus[1] * cpus[2]
            try:
                temperature = cfg.ORIG_CFG.temperature.val[0][0]
            except:
                try:
                    temperature = cfg.ORIG_CFG.temperature.val
                except:
                    temperature = 300.

        self.p(Elements, "<font color='#888888'>Jobname</font>:  %s" % job_name)
        self.p(Elements,
               "<font color='#888888'>Entry title</font>:  %s" % entry_title)
        Elements.append(platypus.Spacer(1, 12))

        temp_str = '%0.1f' % temperature
        sim_table = [[
            'CPU #', 'Job Type', 'Ensemble', 'Temp. [K]', 'Sim. Time [ns]',
            '# Atoms', '# Waters', 'Charge'
        ]]
        sim_table.append([
            ncpus, job_type, ensemble, temp_str, simulation_time, total_atoms,
            self.water_num, total_charge
        ])

        width = 8 * [.9 * inch]
        gray = rlcolors.Color(old_div(136., 256), old_div(136., 256),
                              old_div(136., 256))
        t = platypus.Table(sim_table,
                           width,
                           style=[('BOTTOMPADDING', (0, 1), (-1, -1), 1),
                                  ('TOPPADDING', (0, 1), (-1, -1), 1),
                                  ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                  ('TEXTCOLOR', (0, 0), (7, 0), gray)])
        Elements.append(t)

        if len(print_disclaimer):
            self.p(Elements, print_disclaimer, fontsize=7, color='gray')

    def processChainSequence(self, protein_st):
        prot_residues = 0
        prot_chains = {}

        for chain in protein_st.chain:
            if not chain.name.strip():
                continue
            st = chain.extractStructure()
            #seq = ""
            rescount = 0
            for res in st.residue:
                prot_residues += 1
                rescount += 1
                #seq+=res.getCode()
            prot_chains[chain.name] = rescount

        if not len(prot_chains):
            rescount = 0
            for res in protein_st.residue:
                prot_residues += 1
            prot_chains['NoChainId'] = prot_residues

        return prot_residues, prot_chains

    def reportAddProteinInfo(self, Elements):
        # this is a list of items
        # explicityly exclude ligand
        if self.protein_asl:
            prot_asl = self.protein_asl
            if self.ligand_asl:
                prot_asl += ' and not ( ' + self.ligand_asl + ' )'
        else:
            prot_asl = '(protein)'
        atom_list = analyze.evaluate_asl(self.cms_st, prot_asl)
        if not len(atom_list):
            return
        Elements.append(platypus.Spacer(1, 12))
        self.p(Elements, "<u>Protein Information</u>")
        Elements.append(platypus.Spacer(1, 12))
        prot_str = self.cms_st.extract(atom_list)
        prot_atoms_total = prot_str.atom_total
        prot_str_noh = prot_str.copy()
        build.delete_hydrogens(prot_str_noh)
        prot_atoms_heavy = prot_str_noh.atom_total
        prot_chg = prot_str.formal_charge

        prot_residues, prot_chains = self.processChainSequence(prot_str)
        prot_table = [[
            'Tot. Residues', 'Prot. Chain(s)', 'Res. in Chain(s)', '# Atoms',
            '# Heavy Atoms', 'Charge'
        ]]

        if prot_chg > 0:
            prot_chg = '+' + str(prot_chg)

        chain_names = str(list(prot_chains)).strip()[1:-1]
        res_numb = str(prot_chains.values()).strip()[1:-1]

        prot_table.append([
            prot_residues, chain_names, res_numb, prot_atoms_total,
            prot_atoms_heavy, prot_chg
        ])

        width = 6 * [1. * inch]

        gray = rlcolors.Color(old_div(136., 256), old_div(136., 256),
                              old_div(136., 256))
        t = platypus.Table(prot_table,
                           width,
                           style=[('BOTTOMPADDING', (0, 1), (-1, -1), 1),
                                  ('TOPPADDING', (0, 1), (-1, -1), 1),
                                  ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                  ('TEXTCOLOR', (0, 0), (5, 0), gray)])
        Elements.append(t)

        Elements.append(platypus.Spacer(1, 12))
        # add image of protein sequence
        sz, temp_seq_file = self.getSequenceViewerImage(prot_str)
        (sx, sy) = self.aspectScale(sz[0], sz[1], 7.0, 5.0)
        seq_img = platypus.Image(temp_seq_file, sx * inch, sy * inch)
        #seq_img = platypus.Image(temp_seq_file)
        Elements.append(seq_img)

    def getSequenceViewerImage(self, prot_str):
        seq_viewer = SequenceViewer()
        seq_viewer.incorporateStructure(prot_str)
        seq_viewer.wrapped = True
        seq_viewer.has_ruler = False
        seq_viewer.display_identity = False
        seq_viewer.display_boundaries = True
        seq_viewer.chain_name_only = True
        seq_viewer.hide_empty_lines = True
        seq_viewer.crop_image = True
        seq_viewer.wrapped_width = 70
        seq_viewer.setColorMode(COLOR_POSITION)
        seq_viewer.addAnnotation(ANNOTATION_SSBOND, remove=True)
        seq_viewer.addAnnotation(ANNOTATION_RESNUM)
        schrod_tmp = fileutils.get_directory_path(fileutils.TEMP)
        temp_seq_file = fileutils.get_next_filename(
            os.path.join(schrod_tmp, "image_tmp.png"), "")
        sz = seq_viewer.saveImage(temp_seq_file)
        return sz, temp_seq_file

    def reportAddLigandInfo(self, Elements):
        atom_list = analyze.evaluate_asl(self.cms_st, self.ligand_asl)
        ligand_st = self.cms_st.extract(atom_list)
        lig_atm_noh, ligand_charge = 0, 0
        lig_natm = len(ligand_st.atom)
        for atom in ligand_st.atom:
            ligand_charge += atom.formal_charge
            if atom.atomic_number != 1:
                lig_atm_noh += 1
        resname = []
        for res in ligand_st.residue:
            resname.append(res.pdbres.strip())
        res = list(set(resname))
        if len(res) == 1 and not len(res[0]):
            ligand_resName = ["None"]
        else:
            ligand_resName = str(res)

        smiles = adapter.to_smiles(ligand_st)

        rotatable_bonds = analyze.get_num_rotatable_bonds(
            ligand_st, max_size=aag.MAX_RING_SIZE)
        # now ligand info goes here
        Elements.append(platypus.Spacer(1, 12))
        self.p(Elements, '<u>Ligand Information</u>')
        Elements.append(platypus.Spacer(1, 12))
        # generate ligand image
        schrod_tmp = fileutils.get_directory_path(fileutils.TEMP)
        temp_file = fileutils.get_next_filename(
            os.path.join(schrod_tmp, "image_tmp.png"), "")
        sz = pl_tools.generate_ligand_2d_image(temp_file,
                                               ligand_st,
                                               ret_size=True)
        (sx, sy) = self.aspectScale(sz[0], sz[1], 3.5, 2.5)
        ligand_img = platypus.Image(temp_file, sx * inch, sy * inch)
        # number of fragments in a ligand
        try:
            ligand_number_of_fragments = len(
                self.interactSurveyPanel.lig_frag_dict)
        except:
            from schrodinger.application.desmond.packages.analysis import \
                get_ligand_fragments
            ligand_number_of_fragments = len(get_ligand_fragments(ligand_st))

        if ligand_charge > 0:
            ligand_charge_str = '+' + str(ligand_charge)
        else:
            ligand_charge_str = str(ligand_charge)

        smiles_style = ParaStyle
        smiles_style.fontSize = 10
        #smiles_style.fontName = 'Courier'
        natoms_str = "%i (total) %i (heavy)" % (lig_natm, lig_atm_noh)
        smiles_p = platypus.Paragraph(self.parseSMILES_str(smiles),
                                      smiles_style)
        lig_table = [["SMILES", smiles_p, ''],
                     ["PDB Name", ligand_resName[1:-1], ligand_img],
                     ["Num. of Atoms", natoms_str],
                     ["Atomic Mass",
                      "%5.3f au" % (ligand_st.total_weight), ''],
                     ["Charge", ligand_charge_str, ''],
                     [
                         "Mol. Formula",
                         analyze.generate_molecular_formula(ligand_st), ''
                     ], ["Num. of Fragments", ligand_number_of_fragments, ''],
                     ["Num. of Rot. Bonds", rotatable_bonds, '']]

        width = [1.5 * inch, 1.5 * inch, 3.8 * inch]
        gray = rlcolors.Color(old_div(136., 256), old_div(136., 256),
                              old_div(136., 256))
        t = platypus.Table(lig_table,
                           width,
                           style=[('SPAN', (2, 1), (2, 7)),
                                  ('SPAN', (1, 0), (2, 0)),
                                  ('VALIGN', (2, 1), (2, 1), 'MIDDLE'),
                                  ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
                                  ('TOPPADDING', (0, 0), (-1, -1), 1),
                                  ('TEXTCOLOR', (0, 0), (0, 7), gray)])
        Elements.append(t)

    def parseSMILES_str(self, smiles):
        length = len(smiles)
        new_str = ''
        prev_idx = 0
        max_length = 66
        if length <= max_length:
            return smiles
        for i in list(range(0, length, max_length))[1:]:
            new_str += smiles[prev_idx:i + 1] + ' '
            prev_idx = i + 1
        if prev_idx < length:
            new_str += smiles[prev_idx:length]
        return new_str

    def reportAddIonsSaltsWaterInfo(self, Elements):
        #waters
        try:
            num_water_molecules = self.water_num
        except:
            # get water molecules
            atom_list = analyze.evaluate_asl(self.cms_st, 'water')
            if atom_list:
                water_st = self.cms_st.extract(atom_list)
                num_water_molecules = water_st.mol_total
            else:
                num_water_molecules = 0
        #ions
        ions_num = analyze.evaluate_asl(self.cms_st, 'ions')
        if len(ions_num):
            Elements.append(platypus.Spacer(1, 12))
            self.p(Elements, "<u>Counter Ion/Salt Information</u>")
            Elements.append(platypus.Spacer(1, 12))
            ions_str = self.cms_st.extract(ions_num)
        else:
            return

        ion_type = []
        ion_number = []
        ion_charge = []
        if ions_str:
            for m in ions_str.molecule:
                mf = analyze.generate_molecular_formula(m)
                if not ion_type.count(mf):
                    ion_type.append(mf)
                    #count number of ions of that type
                    ion_number.append(0)
                    #get charge of the molecules
                    m_st = m.extractStructure()
                    ion_charge.append(m_st.formal_charge)
                ion_number[ion_type.index(mf)] += 1

        # counter ions/salt
        ion_items = [['Type', 'Num.', 'Concentration [mM]', 'Total Charge']]
        for i in zip(ion_type, ion_number, ion_charge):
            charge = i[1] * i[2]
            chrg_str = str(charge)
            if charge > 0:
                chrg_str = '+' + chrg_str
            # units: mMol
            concentration = 0.0
            if num_water_molecules > 0:
                concentration = float(
                    i[1]) / (num_water_molecules * 55.) * 1000.
            ion_items.append(
                [i[0], i[1],
                 '%5.3f' % (concentration * 1000), chrg_str])

        width = [0.6 * inch, 0.6 * inch, 1.5 * inch, 1. * inch]
        gray = rlcolors.Color(old_div(136., 256), old_div(136., 256),
                              old_div(136., 256))
        t = platypus.Table(ion_items,
                           width,
                           style=[('BOTTOMPADDING', (0, 1), (-1, -1), 1),
                                  ('TOPPADDING', (0, 1), (-1, -1), 1),
                                  ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
                                  ('TEXTCOLOR', (0, 0), (3, 0), gray)])
        Elements.append(t)

    def reportAddPageSpecific(self, Elements, schrod_tmp):
        if self.panel == 'pl_interact_survey':
            self.interactSurveyPanel.generateReport(self, Elements, schrod_tmp)
        elif self.panel == 'hbond_explorer':
            self.hbondExplorerPanel.generateReport(self, Elements)
        elif self.panel == 'sbridge_explorer':
            self.sbExplorerPanel.generateReport(self, Elements)
        elif self.panel == 'hphobe_explorer':
            self.hpExplorerPanel.generateReport(self, Elements)

    def cleanupReportTempFiles(self, tmp_dir):
        """ cleanup temporary PNG files in the schrodinger tmp dir
        that start with image_tmp* """
        filelist = [f for f in os.listdir(tmp_dir) if \
                             f.startswith("image_tmp") or \
                             f.endswith(".eaf")]
        for f in filelist:
            full_f = os.path.join(tmp_dir, f)
            if os.path.exists(full_f):
                os.remove(full_f)

    def generateReport(self, report_dir):
        pdf_file = self.eaf_fn.replace(".eaf", ".pdf")
        pdf_file = os.path.join(report_dir, os.path.split(pdf_file)[1])
        self.generateReportFile(pdf_file)

    def generateReportFile(self, pdf_file):
        """
        Write report PDF file.

        :param pdf_file: name of report PDF file
        :type pdf_file: str
        """
        if not self.cms_st:
            self.warning(
                self.qw, "Load Data First",
                "Analysis data must be present before a report can be "
                "generated.")
            return

        with wait_cursor:
            # generate first page of report here
            Elements = []
            self.reportAddTopImageAndText(Elements)
            self.reportAddProteinInfo(Elements)

            if self.ligand_asl:
                self.reportAddLigandInfo(Elements)

            self.reportAddIonsSaltsWaterInfo(Elements)

            Elements.append(platypus.PageBreak())

            schrod_tmp = fileutils.get_directory_path(fileutils.TEMP)
            self.reportAddPageSpecific(Elements, schrod_tmp)

            # build document
            doc = platypus.SimpleDocTemplate(pdf_file)
            doc.title = u"Simulation Interactions Diagram by Schrodinger Inc."
            doc.author = "Dmitry Lupyan"
            doc.leftMargin = 30.
            doc.rightMargin = 20.
            doc.topMargin = 10.
            doc.bottomMargin = 20.
            doc.build(Elements, canvasmaker=NumberedCanvas)
            self.cleanupReportTempFiles(schrod_tmp)

            return

    def header(self,
               Elements,
               txt,
               style=HeaderStyle,
               klass=platypus.Paragraph,
               sep=0.3):
        s = platypus.Spacer(0.2 * inch, sep * inch)
        Elements.append(s)
        style.alignment = 1
        style.fontName = 'Helvetica-Bold'
        style.fontSize = 18
        para = klass(txt, style)
        Elements.append(para)

    def createItems(self, Elements, lines):
        for line in lines:
            txt = '- ' + str(line)
            self.p(Elements, txt)

    def p(self, Elements, txt, fixed=False, fontsize=11, color='black'):
        style = ParaStyle
        if fixed:
            style.fontName = 'Courier'
        else:
            style.fontName = 'Helvetica'
        style.fontSize = fontsize
        style.textColor = color
        #style.bulletText = '*'
        para = platypus.Paragraph(txt, style)
        Elements.append(para)

    def aspectScale(self, ix, iy, bx, by):
        """Scale image to fit into box (bx, by) keeping aspect ratio"""
        if ix > iy:
            # fit to width
            scale_factor = old_div(bx, float(ix))
            sy = scale_factor * iy
            if sy > by:
                scale_factor = old_div(by, float(iy))
                sx = scale_factor * ix
                sy = by
            else:
                sx = bx
        else:
            # fit to height
            scale_factor = old_div(by, float(iy))
            sx = scale_factor * ix
            if sx > bx:
                scale_factor = old_div(bx, float(ix))
                sx = bx
                sy = scale_factor * iy
            else:
                sy = by

        return (sx, sy)

    def _importTrajectoryIntoMaestro(self, traj_filename: str):
        """
        Import trajectory into Maestro if maestro is available.

        :param traj_filename: trajectory file
        """
        if maestro:
            # Enclose file name in quotes to allow paths with spaces
            maestro.command(f'entryimport format=cms "{traj_filename}"')


class customDelegate(table.StructureDataViewerDelegate):
    """
    Re-implemented so that paintStructure would have access to struct, so
    that the image could have the residue from the ST printed on it.
    """

    def _paint(self, painter, option, index, passive=False):
        """ See table.py """
        struct = index.data(QtCore.Qt.DisplayRole)
        if struct is None:
            #Skip painting if there's no structure for this cell
            return
        elif self.isStructure(struct):
            # This cell contains a 2D chemical structure, try and grab it from
            # the cache.
            pic = self.picture_cache.get(struct)
            if not pic:
                if passive and self.generate_one_structure:
                    passive = False
                    self.generate_one_structure = False

                if passive:
                    table.GenericViewerDelegate._paint_passive(
                        self, painter, option, index)
                else:
                    # We need to plot a structure, and no structure is
                    # available, generate one and store it in the cache
                    pic = self.generatePicture(struct)
                    self.picture_cache.store(struct, pic)
                    self.paintStructure(painter, option, pic, struct)
            else:
                self.paintStructure(painter, option, pic, struct)
        else:
            self.paintCell(painter, option, struct)

    def paintStructure(self, painter, option, pic, struct):
        """
        Adds a bit of padding all around the cell, then passes the data on to
        the proper drawing routine.

        :type painter: QtGui.QPainter object
        :param option: Appears to be the part of the gui that contains the cell

        :type pic: QPicture
        :param pic: the picture to be painted
        """

        r = option.rect
        padding_factor = 0.04
        r.setLeft(int(option.rect.left() +
                      padding_factor * option.rect.width()))
        r.setRight(
            int(option.rect.right() - padding_factor * option.rect.width()))
        r.setTop(int(option.rect.top() + padding_factor * option.rect.height()))
        r.setBottom(
            int(option.rect.bottom() - padding_factor * option.rect.height()))
        swidgets.draw_picture_into_rect(painter,
                                        pic,
                                        r,
                                        max_scale=self.max_scale)
        painter.drawText(r.left(), r.bottom(), "Residue: %s" % \
                struct.atom[1].pdbres)


# This class is used by reportlab to generate page numbering in
# 'page N of M' format
class NumberedCanvas(canvas.Canvas):

    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        """add page info to each page (page x of y)"""
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.drawPageNumber(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def drawPageNumber(self, page_count):
        self.setFont("Helvetica", 8)
        self.drawRightString(8 * inch, 0.3 * inch,
                             "Page %d of %d" % (self._pageNumber, page_count))
        s = "Schrodinger Inc. Report generated " + datetime.now().strftime(
            "%m-%d-%Y %H:%M")
        self.drawString(0.8 * inch, 0.3 * inch, s)


def show_panel(inp_file=None):
    """Opens a new panel and avoids garbage collection"""

    global panels
    try:
        panels
    except NameError:
        panels = []
    panels.append(EventAnalysisPanel(inp_file=inp_file))


def validate_inp(inp):
    # return if valid filename is used in the arguments.  No filename is
    # valid too, in which case a GUI interface will open up
    if inp is None:
        return True
    if inp.endswith(('eaf', 'cms')):
        return True
    return False


__doc__ = """
This program will set up manually setup SID-related simulations.

Written by Dmitry Lupyan

$Revision:0.2 $
Copyright Schrodinger, LLC. All rights reserved.
"""


def setupAtomProperties(cms_st):
    '''
    Setup original atom indeces, so when we extract structures, we can
    map their atoms in the original atoms
    '''
    for a in cms_st.atom:
        a.property[ORIGINAL_INDEX] = a.index
        a.property[ORIGINAL_MOLECULE_NUMBER] = a.molecule_number


def parse_arguments(argv):
    """
    Parse the command-line arguments.

    :param argv: list of command line arguments.
    :type argv: list

    :return: argument namespace
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Event Analysis')
    subparsers = parser.add_subparsers(dest='command')
    # GUI mode
    gui_parser = subparsers.add_parser(GUI, help='Show event analysis GUI')
    gui_parser.add_argument('input_file', help='EAF or CMS input file or None')
    # Analyze mode
    analyze_parser = subparsers.add_parser(ANALYZE, help='Run in analyze mode')
    analyze_parser.add_argument('cms_file', help='Input *.cms file.')
    analyze_parser.add_argument('-prot',
                                '-p',
                                '-protein',
                                type=str,
                                dest='prot',
                                default='(protein)',
                                metavar='<asl expr>',
                                help='Specify protein ASL selection or "none".')
    analyze_parser.add_argument(
        '-lig',
        '-l',
        '-ligand',
        type=str,
        dest='lig',
        default='auto',
        metavar='<asl expr>',
        help='Specify ligand ASL string , "none" or "auto"')
    analyze_parser.add_argument('-out',
                                '-o',
                                type=str,
                                dest='basename',
                                default=None,
                                metavar='basename',
                                help='The name of the output')
    # Report mode
    report_parser = subparsers.add_parser(
        REPORT, help='Generate reports from command line')
    report_parser.add_argument('eaf_file', help='Input *.eaf file')
    report_parser.add_argument('-pdf',
                               dest='pdf_file',
                               metavar='<pdf file>',
                               help='Name of report pdf file.')
    report_parser.add_argument(
        '-data',
        action='store_true',
        help='Generate text data files in the data directory')
    report_parser.add_argument(
        '-plots',
        action='store_true',
        help='Generate plot images in the data directory')
    report_parser.add_argument('-data_dir',
                               dest='data_directory',
                               metavar='<data directory>',
                               default='.',
                               help='Directory for text data and plot files')

    args = parser.parse_args(argv)
    if args.command is None:
        args.command = GUI
        args.input_file = None

    return args


class ReportGenerationTask(tasks.ComboSubprocessTask):
    """
    Generate a SID report as a PDF file,
    as well as data (.dat files) and plots.
    """

    class Input(CompoundParam):
        eaf_file: tasks.TaskFile = None
        report_file_name: str = None
        write_dat_files: bool = False
        write_plot_files: bool = False

    class Output(CompoundParam):
        report_file: tasks.TaskFile = None

    @tasks.preprocessor
    def _checkInputEAFFileValid(self):
        """
        Check that the input eaf file exists and
        is of the correct type.
        """
        return check_input_file_is_valid(self.input.eaf_file, EAF_EXT)

    @tasks.preprocessor
    def _checkReportFileNameValid(self):
        """
        Check that the filename for the report is valid
        or set it if it is unset.
        """
        task_dir = self.getTaskDir()
        if not self.input.report_file_name:
            basename = os.path.basename(self.input.eaf_file).replace(
                EAF_EXT, PDF_EXT)
            self.input.report_file_name = os.path.join(task_dir, basename)
            return True
        elif self.input.report_file_name.endswith(PDF_EXT):
            return True
        return False, make_preprocessor_failure_message(
            self.input.report_file_name, PDF_EXT)

    def mainFunction(self):
        """
        Creates a hidden version of the panel and uses it to generate report data
        """
        ea = EventAnalysisPanel(test_mode=True)
        ea.loadEAF(self.input.eaf_file)
        pdf_file = self.input.report_file_name
        ea.generateReportFile(pdf_file)
        task_dir = self.getTaskDir()
        if self.input.write_dat_files:
            ea.interactSurveyPanel.export_data(task_dir)
        if self.input.write_plot_files:
            ea.interactSurveyPanel.export_plots(task_dir)
        self.output.report_file = pdf_file


def _run_report_generation_task(args):
    """
    Run the ReportGenerationTask.
    """
    report_gen_task = ReportGenerationTask()
    report_gen_task.input.eaf_file = args.eaf_file
    report_gen_task.input.report_file_name = args.pdf_file
    report_gen_task.input.write_dat_files = args.data
    report_gen_task.input.write_plot_files = args.plots
    if args.data_directory:
        report_gen_task.specifyTaskDir(args.data_directory)
    report_gen_task.runInProcess()


def _run_eaf_generation_task(args):
    """
    Run the EAFGenerationTask.
    """
    eaf_gen_task = EAFGenerationTask()
    eaf_gen_task.input.cms_file = args.cms_file
    eaf_gen_task.input.protein_asl = args.prot
    eaf_gen_task.input.ligand_asl = args.lig
    eaf_gen_task.input.out_file_name = args.basename
    eaf_gen_task.runInProcess()


class EventAnalysisTask(tasks.ComboSubprocessTask):
    """
    Combined task that runs EAFGenerationTask, EAFAnalysisTask, and ReportGenerationTask.
    Input is a trajectory, protein ASL, and ligand ASL.
    Outputs the PDF report file.
    """

    class Input(CompoundParam):
        cms_file: tasks.TaskFile = None
        protein_asl: str = DEFAULT_PROT_ASL
        ligand_asl: str = DEFAULT_LIG_ASL
        write_dat_files: bool = False
        write_plot_files: bool = False

    class Output(CompoundParam):
        report_file: tasks.TaskFile = None

    @tasks.preprocessor
    def _checkInputEAFFileValid(self):
        """
        Check that the input eaf file exists and
        is of the correct type.
        """
        return check_input_file_is_valid(self.input.eaf_file, EAF_EXT)

    @tasks.preprocessor
    def _checkReportFileNameValid(self):
        """
        Check that the filename for the report is valid
        or set it if it is unset.
        """
        task_dir = self.getTaskDir()
        if not self.input.report_file_name:
            basename = os.path.basename(self.input.eaf_file).replace(
                EAF_EXT, PDF_EXT)
            self.input.report_file_name = os.path.join(task_dir, basename)
            return True
        elif self.input.report_file_name.endswith(PDF_EXT):
            return True
        return False, make_preprocessor_failure_message(
            self.input.report_file_name, PDF_EXT)

    def mainFunction(self):
        """
        Creates a hidden version of the panel and uses it to generate report data
        """
        # Create an event analysis (.eaf) file from the input CMS:
        print('Generating EAF file...')
        eaf_gen_task = EAFGenerationTask()
        eaf_gen_task.input.cms_file = self.input.cms_file
        eaf_gen_task.input.ligand_asl = self.input.ligand_asl
        eaf_gen_task.input.out_file_name = 'event_analysis'
        eaf_gen_task.specifyTaskDir(None)  # Run in same directory
        eaf_gen_task.runInProcess()  # Run in same process

        eaf_file = eaf_gen_task.output.eaf_file
        assert os.path.isfile(eaf_file)

        print('Running analyze_simulation.py...')
        eaf_analysis_task = EAFAnalysisTask()
        eaf_analysis_task.input.cms_file = self.input.cms_file
        eaf_analysis_task.input.trajectory = topo.find_traj_path_from_cms_path(
            self.input.cms_file)
        eaf_analysis_task.input.out_file_name = "analysis_task_output.eaf"
        eaf_analysis_task.input.eaf_file = eaf_file
        eaf_analysis_task.start()
        eaf_analysis_task.wait(check=True)

        out_eaf_file = eaf_analysis_task.output.eaf_file
        assert os.path.isfile(out_eaf_file)

        print('Generating report...')
        report_gen_task = ReportGenerationTask()
        report_gen_task.input.eaf_file = out_eaf_file
        report_gen_task.input.report_file_name = "test_report.pdf"
        report_gen_task.input.write_dat_files = self.input.write_dat_files
        report_gen_task.input.write_plot_files = self.input.write_plot_files
        report_gen_task.specifyTaskDir(None)  # Run in same directory
        report_gen_task.runInProcess()  # Run in same process

        report_file = report_gen_task.output.report_file
        assert os.path.isfile(report_file)
        self.output.report_file = report_file


if __name__ == '__main__':

    args = parse_arguments(sys.argv[1:])
    # ANALYZE and REPORT don't require $DISPLAY. If $DISPLAY is unset,
    # get_application will create an offscreen QApplication
    qapplication.get_application()
    if args.command == GUI:
        inp_file = args.input_file
        if validate_inp(inp_file):
            global panels
            show_panel(inp_file=inp_file)
            panels[0].app.exec()
    elif args.command == ANALYZE:
        _run_eaf_generation_task(args)
    elif args.command == REPORT:
        _run_report_generation_task(args)
