from IO.FileManager import FileManager

fm = FileManager()
try:
    mealy_machine = fm.loadMealyFromDotFile("LoesTarget.dot")
    print(mealy_machine)
except Exception as e:
    print(f"Error: {e}")