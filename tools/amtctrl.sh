#!/bin/bash

# Rechner btcipai61 - btcipai73 im Raum 1.37 steuern.

# Standardvariablen:
hostname=btcipai
start_addr=0
end_addr=0
cmd="" # Status des Rechners ausgeben, falls kein powerup oder powerdown verlangt wird
parallel="&" # Setze ein & für parallele Ausführung des Befehls, leer lassen falls nicht

function helptext () {
    echo "Befehle:"
    echo "-s  --start            Erster Rechner fuer Befehl (z.B. 61)"
    echo "-e  --end              Letzter Rechner fuer Befehl (z.B. 73)"
    echo "-w  --powerup          Rechner im Adressbereich aufwecken (via AMT)"
    echo "    --halt --powerdown Rechner herunterfahren"
    echo "-h  --help             Hilfe anzeigen"
    echo "-o  --omit             Rechner auslassen (durch Komma getrennt)"
    echo "    --status           Status des Rechners ausgeben"
    exit 0
}

function exec_cmd () {
  for ((i=start_addr;i<=end_addr;i++)); do
    # Aktuellen Rechner auslassen?
      for element in ${omitips[*]}; do
        if (($i==$element)); then
          # Mit Schleife aussen weiter machen
          continue 2
        fi
      done
        
      amt_cmd="amttool $hostname$i $cmd $parallel"
      echo "Executing $amt_cmd"
      eval "$amt_cmd" || error
    done
}

function error () {
    echo "ERROR: Befehl war $cmd"
    echo ""
    cleanup
    stty sane # Konsole zuruecksetzen
    exit 6
}

function parse () {
    # Rechner auslassen (Array durch Kommata getrennt)
    IFS=","
    omitips=($1)
    echo "Rechner auslassen: " ${omitips[*]}
}

function chk_switches () {
    # Switches auswerten
    if (("$#" == 0)); then helptext ; fi
    while [ "$#" -gt "0" ]; do
      case "$1" in
        -s|--start)
          shift; start_addr=$1
          echo "Erster Rechner: $start_addr"
          ;;
        -e|--end)
          shift; end_addr=$1
          echo "Letzter Rechner: $end_addr"
          ;;
        -o|--omit)
          shift; parse $1
          ;;
        -r|--room)
          start_addr=61
          end_addr=73
          ;;
        --halt|--powerdown)
          # Faehrt Rechner herunter
          cmd='powerdown'
          ;;
        -w|--powerup)
          # Rechnernamen
          cmd='powerup'
          ;;
        --) break
          ;;
      esac
        shift
      done
}

function validate () {
    # Plausibilitaetstests
    if(($start_addr > $end_addr)); then
    echo "Startadresse groesser als Endadresse!"
    helptext
    exit 1
    fi

    if(($start_addr < 61)); then
    echo "Ungueltige Startadresse!"
    helptext
    exit 2
    fi

    if(($end_addr > 73)); then
    echo "Ungeutlige Endadresse!"
    helptext
    exit 3
    fi
}

chk_switches "$@"   # Uebergebene Parameter speichern
validate            # Parameter pruefen
exec_cmd            # Befehl ausfuehren
exit 0
