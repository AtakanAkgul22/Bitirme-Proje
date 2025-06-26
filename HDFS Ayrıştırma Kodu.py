#HDFS AYRIŞTIRMA KODU DRAİN.PY DOSYASININ EN ALTINA YAPIŞTIRDIĞIM KOD
if __name__ == "__main__":
    parser = LogParser(
        log_format="<Date> <Time> <Pid> <Level> <Component>: <Content>",
        girdi="../../../loghub/HDFS/",
        çıktı="../../../yapilandirilmis/HDFS/"
    )
    parser.parse("HDFS_2k.log")