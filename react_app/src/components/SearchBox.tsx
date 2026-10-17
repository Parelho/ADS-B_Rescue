import { useId, useMemo, useState } from "react";

interface Props {
  options: string[];
  onSelect: (value: string) => void;
}

/** ICAO autocomplete: case-insensitive "contains" match, like the QCompleter. */
export default function SearchBox({ options, onSelect }: Props) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const listId = useId();

  const matches = useMemo(() => {
    const q = query.trim().toUpperCase();
    return q ? options.filter((o) => o.includes(q)) : [];
  }, [query, options]);

  const choose = (value: string) => {
    setQuery(value);
    setOpen(false);
    onSelect(value);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setOpen(true);
      setActive((i) => Math.min(i + 1, matches.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && open && matches[active]) {
      e.preventDefault();
      choose(matches[active]);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  };

  return (
    <div className="relative w-[300px] max-w-full">
      <input
        type="text"
        className="h-11 w-full rounded-lg border-2 border-gray-300 bg-white px-3 text-[15px] text-black outline-none focus:border-blue-500 focus:bg-gray-50"
        placeholder="Digite o código ICAO"
        value={query}
        role="combobox"
        aria-expanded={open && matches.length > 0}
        aria-controls={listId}
        aria-autocomplete="list"
        onChange={(e) => {
          setQuery(e.target.value);
          setActive(0);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={onKeyDown}
      />
      {open && matches.length > 0 && (
        <ul
          className="absolute inset-x-0 top-[calc(100%+4px)] z-[1000] my-0 max-h-[260px] list-none overflow-y-auto rounded-lg border border-gray-300 bg-white py-1 shadow-[0_4px_12px_rgba(0,0,0,0.12)]"
          id={listId}
          role="listbox"
        >
          {matches.map((m, i) => (
            <li
              key={m}
              role="option"
              aria-selected={i === active}
              className={`cursor-pointer px-3 py-1.5 ${i === active ? "bg-blue-50" : ""}`}
              // mousedown fires before the input's blur closes the list
              onMouseDown={(e) => {
                e.preventDefault();
                choose(m);
              }}
              onMouseEnter={() => setActive(i)}
            >
              {m}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
