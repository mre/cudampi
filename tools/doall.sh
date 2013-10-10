#!/bin/bash
# doall
# Universitaet Bayreuth
# Skript das einen Unix-Befehl auf mehreren Rechnern (parallel) ausfuehrt.
# Copyright (C) 2009 Matthias Endler
# Version 0.4

# Bash-Einstellungen
execution=""         # wird execution auf "&" gesetzt, so wird der Befehl parallel ausgefuehrt
c1="\E[37;44m\033[1m"     # Farbe Blau Anfang
c2="\033[0m"        # Farbe Blau Ende
sshops="-CXY -o BatchMode=yes -o ConnectTimeout=8" # SSH Optionen
kill=0            # Falls Wert 1, wird die Ausführung abgebrochen
trap "kill=1" SIGINT SIGTERM # Befehle abbrechen bei CTRL^C


# Standardvariablen:
hostname=btcipai
start_addr=0
end_addr=0
args=""
timeout=2 # Anzahl Pings
remotecmd="ssh $sshops $hostname$i"


# Umgebungsvariablen fuer das Verteilen von Images
run="NO"
calc_md5sum=0        # md5 Checksumme nicht vergleichen
RECEIVER_PATH=/usr/sbin
SENDER_PATH=/usr/sbin
receiver_ops="--nokbd "
udp_logfile=udplog    # Pfad zur Logdatei von udp-sender
sender_ops="--interface bond0 --full-duplex --min-receivers $clientnumber --log $udp_logfile --file"  # --autostart $clientnumber --max-wait 500000
file="."         # Image

function helptext () {
    echo "Befehle:"
    echo "-c  --command          Befehl (in Anfuehrungszeichen)"
    echo "-d  --distribute       Imagedatei auf Rechner schreiben"
    echo "-e  --end              Letzter Rechner fuer Befehl"
    echo "    --halt --shutdown  Rechner herunterfahren"
    echo "-h  --help             Hilfe anzeigen"
    echo "-o  --omit             Rechner auslassen (durch Komma getrennt)"
    echo "-os --system           Betriebssystem ausgeben"
    echo "    --parallel         Befehl auf allen Rechnern parallel ausfuehren"
    echo "-p  --ping             Angegebene Rechner pingen (geht auch mit -c ping)"
    echo "    --reboot           Rechner neu starten"
    echo "-r  --room             Befehl fuer ganzen Raum ausfuehren [103|137|150|201]"
    echo "-s  --start            Erster Rechner fuer Befehl (z.B. 22)"
    echo "-w  --wake             Rechner im Adressbereich aufwecken (Wake-on-LAN)"
    echo ""
    echo "Beispiele:"
    echo "doall -p -r all                  Pingt alle Rechner an der Uni"
    echo "doall -w -s 12 -e 44 -o 5,2,33   Startet Rechner 12 bis 44 ausser 5, 2 und 33"
    echo "doall -s 4 -c \"uptime && who\"    Fuehrt auf Rechner 4 die Befehle 'uptime' und 'who' aus"
    echo "doall -d IMAGEFILE -r 201        Die Datei IMAGEFILE auf alle Rechner im Raum 2.01 schreiben"
    exit 0
}

function exec_cmd () {
    for ((i=start_addr;i&lt;=end_addr;i++)); do

    # Aktuellen Rechner auslassen?
    for element in ${omitips[*]}; do
    if (($i==$element)); then
    # Mit Schleife aussen weiter machen
    continue 2
    fi
    done

    # CTRL^C gedrueckt?
    if (($kill==1)); then
    error # Fehlermeldung und beenden
    fi

    # Hostnamen farbig ausgeben
    echo -ne $c1$hostname$i":"$c2
    tput sgr0 # Farbe zuruecksetzen
    echo -ne " "

    # Befehl ausfuehren
    eval "$fullcmd $execution" || error
    done
}

function error () {
    echo "ERROR: Befehl war $fullcmd"
    echo ""
    cleanup
    stty sane # Konsole zuruecksetzen
    exit 6
}

function cleanup () {
    case "$command" in
    distribute)
    SuSEfirewall2 start
    ;;
    os)
    echo -e "\nUebersicht (L=Linux, W=Windows, X=Keine Antwort)"
    stepping=4
    for ((i=start_addr;i&lt;=end_addr;i=i+$stepping)); do
    echo "${grid[$i]} ${grid[(($i+1))]} ${grid[(($i+2))]} ${grid[(($i+3))]}"
    done
    ;;
    esac
}

function parse () {
    # Rechner auslassen (Array durch Kommata getrennt)
    IFS=","
    omitips=($1)
    echo "Rechner auslassen: " ${omitips[*]}
}

function os_check () {
    # Betriebssystem der Clients ermitteln
    output=`ssh $sshops $hostname$i uname 2&gt;&1`

    # Checks fuer die einzelnen Systeme
    lintest=$(echo "$output" | grep -c "Linux")              # Linux-Check
    wintest=$(echo "$output" | grep -c "@")             # Windows-Check fuer Server
    wintes2=$(echo "$output" | grep -c "verification failed")     # Windows-Check fuer normale Benutzer
    refused=$(echo "$output" | grep -c "Connection refused")    # Rechenzentrum?
    noroute=$(echo "$output" | grep -c "No route")             # Can't find host on network
    nopermission=$(echo "$output" | grep -c "Permission denied")    # Kein Zugriffsrecht
    timeout=$(echo "$output" | grep -c "timed out")         # Timeout

    if [ "$lintest" -ne 0 ]; then
    echo "Linux"
    grid[$i]="L"
    elif [ "$wintest"  -ne 0 ]; then
    echo "Windows"
    grid[$i]="W"
    elif [ "$wintes2"  -ne 0 ]; then
    echo "Windows"
    grid[$i]="W"
    elif [ "$timeout" -ne 0 ]; then
    echo "TIMEOUT"
    grid[$i]="X"
    elif [ "$refused" -ne 0 ]; then
    echo "Rechenzentrum (CONNECTION REFUSED)"
    Grid[$i]="X"
    elif [ "$noroute" -ne 0 ]; then
    echo "Off (NO ROUTE TO HOST)"
    grid[$i]="X"
    elif [ "$nopermission" -ne 0 ]; then
    echo "PERMISSION DENIED"
    grid[$i]="X"
    else
    echo "ERROR: $output"
    grid[$i]="X"
    fi
}

function md5sum () {
    matches=0 # Wird auf 1 gesetzt falls md5 Summe korrekt
    echo "Berechne md5-Summe auf SERVER. Bitte warten..."
    srvchk=$(md5sum $file | awk '{ print $1}')
    echo "MD5-Checksumme auf Server: $srvchk"
}

function distribute() {
    # Funktion zum Verteilen und Schreiben von Images auf Clients
    if [ "$run" != "YES" ]; then

    # Imagedatei auswaehlen
    useimage="N"
    while [ "$useimage" != "y" ]; do
    if [ -f "$file" ]; then
    echo "Dateiname fuer Image: $file"
    echo "Dieses Image verwenden (ENTER fuer Abbruch)? [y|n]"
    read -s -n 1 useimage
    if [ -z "$useimage" ]; then exit 6; fi
    if [ "$useimage" == "Y" ]; then useimage="y"; fi
    if [ "$useimage" != "y" ]; then file="."; fi
    else
    echo "Pfad und Name des Images angeben (ENTER fuer Abbruch, Vorschlag mit v):"
    read file
    if [ -z "$file" ]; then exit; fi
    if [ "$file" == "v" ]; then
    # Image automatisch vorschlagen:
    file=$(ls -tr *.img.gz | tail -n 1)
    fi
    fi
    done


    # Verzeichnis, in das das Image gespeichert wird
    tmpdir_chk=1
    while [ "$tmpdir_chk" -ne 0 ]; do
    echo "Wohin soll das Image auf den Clients gespeichert werden?"
    echo "z.B.:  /mnt/sda7"
    read directory
    tmpdir_chk=$(ssh $sshops $hostname$i "test -d $directory  && test -w $directory && echo 0 || echo 1" || echo 1)
    if [ "$tmpdir_chk" -eq 0 ]; then
    echo "Verzeichnis kann verwendet werden"
    else
    echo "Verzeichnis ist ungueltig"
    fi
    done

    # Imagetyp Win oder Linux Image?
    echo "Typ angeben [ 1 fuer Windows | 0 fuer Linux ]"
    read -s -n 1 imagetype
    if [ "$imagetype" -ne 0 ] && [ "$imagetype" -ne 1 ]; then
    echo "Fehlerhafte Eingabe"
    exit 2
    fi


    # Methode auswaehlen
    echo "1) Image nur uebertragen"
    echo "2) Image uebertragen und dann schreiben (langsam, sicher)"
    echo "3) Image direkt schreiben (schnell, unsicher)"
    echo "4) Image auf Client zurueckspielen"
    echo "Auswahl mit [1|2|3|4] (Abbruch mit anderem Zeichen):"
    read -s -n 1 method

    writecmd=""
    case $method in
    1)
    # Nur uebertragen, nicht schreiben
    receiverfile="--file $directory/$file"
    ;;
    2)
    # Uebertragen dann schreiben
    if [ "$imagetype" -eq 0 ]; then # Imagetyp: LINUX
    receiverfile="--file $directory/$file"
    writecmd="read; tar -xzf $directory/$file -C $device"
    elif [ "$imagetype" -eq 1 ]; then # Imagetyp WINDOWS
    receiverfile="--file $directory/$file"
    writecmd="gunzip -c $directory/$file | ntfsclone --restore-image --overwrite $device -"
    fi
    ;;
    3)
    # Direkt schreiben
    if [ "$imagetype" -eq 0 ]; then # Imagetyp: LINUX
    # FIXME
    #writecmd="| tar -xzf $directory/$file -C $device"
    writecmd="| tar -xzf -C $device"
    elif [ "$imagetype" -eq 1 ]; then # Imagetyp WINDOWS
    receiver_ops="$receiver_ops -p \"gunzip -c -\""
    writecmd="| ntfsclone --restore-image --overwrite $device -"
    fi
    ;;
    4)
    # Image ruecksichern
    methodcmd=""
    ;;
    *)
    exit 0;
    ;;
    esac

    # Festplatte fuer Image
    #FIXME prüft auf Server!
    if [ "$method" -ne 1 ]; then
    testdir=0
    while [ "$testdir" -ne 1 ]; do
    echo "Auf welche Festplatte soll geschrieben werden (ENTER fuer Abbruch)?"
    read device
    if [ -z "$device" ]; then exit 6; fi
    testdir=$(mount | grep $device | grep -c rw)
    done

    fi

    # Zusaetzliche Befehle fuer Windows
    if [ "$imagetype" -eq 1 ] && [ "$method" -ne 1 ]; then
    mount_fsdir="ntfs-3g $device /mnt/sda2"
    setAutoLogon="sed -i s/AutoLogonCount=./AutoLogonCount=0/g sysprep.inf"
    #setcmd="&& $mount_fsdir && $setHostname && $setAutoLogon"
    fi

    # FIXME Ausgelassene Rechner beruecksichtigen
    clientnumber=$(($end_addr-$start_addr+1))


    # Sicherheitsabfrage
    if [ "$method" -ne 1 ]; then
    echo "ACHTUNG! Dateien werden ueberschrieben! Befehl wirklich ausfuehren?"
    echo "Bestaetigen mit YES, Abbrechen mit ENTER:"
    read -n 3 run
    case "$run" in
    YES)
    ;;
    "")
    exit 1
    ;;
    *)
    echo "Bestaetigen mit YES, Abbrechen mit ENTER:"
    ;;
    esac
    fi
    run="YES" # Falls das Image nur uebertragen wird
    fi

    # udp-receiver auf clients starten
    # Die Nummern fuer den Hostnamen müssen zweistellig sein:
    case "$i" in
    [0-9])
    z=0$i
    ;;
    *)
    z=$i
    ;;
    esac
    setcmd="&& $mount_fsdir && cd /mnt/sda2/sysprep/ && sed -i s/btcipai../$hostname$z/g sysprep.inf && $setAutoLogon"
    imagecmd="ssh $sshops $hostname$i 'hostname; killall -9 udp-receiver 2&gt; /dev/null; SuSEfirewall2 stop; sleep 4;\
    udp-receiver $receiver_ops $receiverfile $methodcmd $writecmd $setcmd'; read"
    echo "FUEHRE AUS: $imagecmd"
    echo "...starte udp-receiver auf $hostname$i..."
    xterm -bg grey -e "$imagecmd" &


    # Sender nach allen Clients starten:
    if(("$i"=="$end_addr")); then
    #SuSEfirewall2 stop
    echo "OK. Starte sender in 10..."
    for ((j=9;j&gt;=1;j--)); do
    echo "$j..."
    # CTRL^C gedrueckt?
    if (("$kill"==1)); then
    error # Fehlermeldung und beenden
    fi
    sleep 1
    done
    udp-sender $sender_ops $file
    fi
    # TODO Evtl. Checksumme auf Client berechnen
    # TODO Zusammenfassung anzeigen
}

function chk_switches () {
    # Switches auswerten
    if (("$#" == 0)); then helptext ; fi
    while [ "$#" -gt "0" ]; do
    case "$1" in
    -c|--command)
    shift; command="$1"
    fullcmd="$remotecmd $command"
    #fullcmd='ssh $sshops $hostname$i $command &gt; /dev/null || echo 1'
    ;;
    -d|--distribute)
    if [ -e !"$RECEIVER_PATH/udp-sender" ]; then
    echo "Programm udp-sender nicht vorhanden"
    exit 1
    fi
    if [ -e !"$RECEIVER_PATH/udp-receiver" ]; then
    echo "Programm udp-receiver nicht vorhanden"
    exit 1
    fi
    fullcmd=distribute
    ;;
    -os|--system)
    fullcmd=os_check
    ;;
    -e|--end)
    shift; end_addr=$1
    echo "Letzter Rechner: $end_addr"
    ;;
    -o|--omit)
    shift; parse $1
    ;;
    --parallel)
    execution="&"
    ;;
    -p|--ping)
    fullcmd='ping -c 1 -q -w $timeout $hostname$i | tail -n 2 | head -n 1'
    ;;
    -r|--room)
    shift;
    case "$1" in
    103 | 1.03)
    start_addr=40
    end_addr=60
    ;;
    137 | 1.37)
    start_addr=61
    end_addr=73
    ;;
    150 | 1.50)
    start_addr=74
    end_addr=75
    ;;
    201 | 2.01)
    start_addr=1
    end_addr=39
    ;;
    all)
    start_addr=1
    end_addr=73
    ;;
    *)
    echo "Raumnummer nicht erkannt"
    helptext
    ;;

    esac
    ;;
    --halt|--shutdown)
    # Faehrt Rechner herunter
    l_shutdown='ssh $sshops $hostname$i halt'
    w_shutdown='su admin-xp -c "ssh Administrator@$hostname$i shutdown -x now"'
    if [ "$os_select" = "linux" ]; then
    fullcmd="$l_shutdown || echo 1"
    elif [ "$os_select" = "windows" ]; then
    fullcmd="$w_shutdown  || echo 1"
    else
    fullcmd="$l_shutdown || $w_shutdown || echo 1"
    fi
    case "$2" in
    linux|Linux)
    echo "Nur Linux herunterfahren"
    os_select="linux"
    shift
    ;;
    windows|Windows)
    echo "Nur Windows herunterfahren"
    os_select="windows"
    shift
    ;;
    *)
    os_select="none"
    ;;
    esac

    ;;
    --reboot)
    # Startet alle Rechner neu
    w_reboot='su admin-xp -c "ssh -o Batchmode=yes -l Administrator $hostname$i reboot now"'
    l_reboot='ssh $sshops $hostname$i reboot'
    if [ "$os_select" = "linux" ]; then
    fullcmd="$l_reboot || echo 1"
    elif [ "$os_select" = "windows" ]; then
    fullcmd="$w_reboot || echo 1"
    else
    fullcmd="$l_reboot || $w_reboot || echo 1"
    fi
    case "$2" in
    linux|Linux)
    echo "Nur Linux neu starten"
    os_select="linux"
    shift
    ;;
    windows|Windows)
    echo "Nur Windows neu starten"
    os_select="windows"
    shift
    ;;
    *)
    os_select="none"
    ;;
    esac
    ;;
    -s|--start)
    shift; start_addr=$1
    echo "Erster Rechner: $start_addr"
    ;;
    -w|--wake)
    # Benutzer ist root?
    if [[ "$UID" -ne 0 ]]; then
    echo "Keine Berechtigung. Nur als root ausfuehrbar"
    exit 6
    fi
    # Macadressen fuer Wake-On-LAN:
    macaddr=(
    # Raum 201 - Rechner 1-39
    0 # Es gibt keinen Rechner btcipai0!
    00:19:99:33:6f:cb 00:19:99:33:80:67 00:19:99:33:81:43 00:19:99:33:83:81 00:19:99:33:81:5d 00:19:99:33:70:33
    00:19:99:33:83:97 00:19:99:33:65:f9 00:19:99:33:80:87 00:19:99:33:83:d1 00:19:99:33:81:27 00:19:99:33:66:73
    00:19:99:33:70:07 00:19:99:33:6f:ab 00:19:99:33:81:6b 00:19:99:33:81:37 00:19:99:33:70:ed 00:19:99:33:80:9d
    00:19:99:33:81:6d 00:19:99:33:70:db 00:19:99:33:81:75 00:19:99:33:81:41 00:19:99:33:66:bf 00:19:99:33:81:45
    00:19:99:33:6f:35 00:19:99:33:66:d5 00:19:99:33:6f:b5 00:19:99:33:70:0d 00:19:99:33:6f:75 00:19:99:33:6f:45
    00:19:99:33:72:6f 00:19:99:33:70:dd 00:19:99:33:70:0b 00:19:99:33:64:db 00:19:99:33:80:83 00:19:99:33:6f:5b
    00:19:99:33:70:11 00:19:99:33:83:75 00:19:99:33:6f:d7
    # Raum 103 - Rechner 40-60
    00:19:99:33:81:a5 00:19:99:33:6f:a9 00:19:99:33:72:d3 00:19:99:33:67:01 00:19:99:33:81:95 00:19:99:33:80:a1
    00:19:99:33:81:33 00:19:99:33:63:f9 00:19:99:33:81:8f 00:19:99:33:80:93 00:19:99:33:6f:79 00:19:99:2a:42:ff
    00:19:99:33:6f:7b 00:19:99:33:81:3b 00:19:99:33:80:85 00:19:99:33:70:df 00:19:99:33:83:cb 00:19:99:2b:d5:0e
    00:19:99:33:81:71 00:19:99:33:80:e7 00:19:99:33:6f:cf
    # Raum 137 - Rechner 61-73
    00:19:99:33:80:7f 00:19:99:33:81:3d 00:19:99:33:81:59 00:19:99:33:80:99 00:19:99:33:80:97 00:19:99:33:6f:47
    00:19:99:33:70:6d 00:19:99:33:80:7d 00:19:99:33:81:77 00:19:99:33:67:0b 00:19:99:33:6f:cd 00:19:99:33:80:6b
    00:19:99:33:80:6d
    # Raum 150 - Rechner 74 und 75
    00:19:99:33:81:35
    00:19:99:33:81:49
    )
    fullcmd='wol ${macaddr[$i]}'
    ;;
    --) break
    ;;
    " " | *)
    # Wurde eine Datei uebergeben?
    if [ -f "$1" ]; then
    if [ $(cat $1 | head -n 2 | grep -c "#DISTRIBUTION FILE") -eq 1 ]; then
    # Konfigurationsdatei ausfuehren
    run="NO"
    . $1
    echo $directory
    else
    file=$1
    fi
    else
    echo "Dateiname oder Switch nicht erkannt: $1"
    helptext
    fi
    ;;
    esac
    shift
    done
}

function validate () {
    # Plausibilitaetstests
    if(($start_addr &gt; $end_addr)); then
    echo "Startadresse groesser als Endadresse!"
    helptext
    exit 1
    fi

    if(($start_addr&lt;=0)); then
    echo "Ungueltige Startadresse!"
    helptext
    exit 2
    fi

    if(($end_addr&lt;=0)); then
    echo "Ungeutlige Endadresse!"
    helptext
    exit 3
    fi

    if [ -z "$fullcmd" ]; then
    echo "Kein Befehl angegeben!"
    echo "Syntax: -c BEFEHL"
    helptext
    exit 4
    fi
    # Einzelne Rechner pingen:
    if(($end_addr==0)); then
    end_addr=$start_addr
    fi
}

chk_switches "$@"        # Uebergebene Parameter speichern
validate        # Parameter pruefen
exec_cmd         # Befehl ausfuehren
cleanup          # Zusammenfassungen, Bereinigung

exit 0

