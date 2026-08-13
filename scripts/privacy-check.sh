#!/bin/sh
set -eu

blocked='\.(docx|pdf|tex|txt)$|(^|/)(raw|private|extracted-text)/'
if git ls-files | grep -E "$blocked"; then
  echo "FAIL: private manuscript material is tracked" >&2
  exit 1
fi

if git grep -n -E 'sureshemc12@gmail\.com|/Users/EB1A|Agentic_Storage_QoS_IEEE_6_Page_Paper\.docx' -- ':!scripts/privacy-check.sh'; then
  echo "FAIL: private identifier or local path is tracked" >&2
  exit 1
fi

echo "PASS: no blocked manuscript files, author email, or local manuscript path tracked"
