import {ReactNode} from 'react';
import {NavLink} from 'react-router-dom';
import {useQuery} from '@tanstack/react-query';
import {api} from '../api';
import {BarChart3,Database,FileText,Home,LayoutDashboard,MessageSquareText,Settings,ShieldCheck,Sparkles,Target} from 'lucide-react';

const nav=[
  ['Home','/',Home],['Workspace','/analyst',MessageSquareText],['Data','/datasets',Database],
  ['Quality','/quality',ShieldCheck],['Models','/experiments',Target],['Charts','/visualizations',BarChart3],['Dashboard','/dashboard',LayoutDashboard],
  ['Reports','/reports',FileText],['Connections','/connections',Settings],
] as const;

export default function Shell({children}:{children:ReactNode}){
  const health=useQuery({queryKey:['health'],queryFn:()=>api<{database:string;storage:string}>('/api/system/health'),refetchInterval:15000,retry:1});
  const ready=health.data?.database==='healthy';
  const recovery=ready&&health.data?.storage!=='PostgreSQL';
  return <div className="min-h-screen bg-[#f5f7fa] text-slate-800">
    <aside className="fixed inset-y-0 left-0 z-40 flex w-[72px] flex-col border-r border-slate-800 bg-[#111b2c] text-white">
      <NavLink to="/" aria-label="DataPilot home" className="grid h-16 place-items-center border-b border-white/10"><span className="grid h-9 w-9 place-items-center rounded-lg bg-teal-400 text-[#10202c]"><Sparkles size={18}/></span></NavLink>
      <nav className="flex-1 space-y-1 px-2 py-3">{nav.map(([label,to,Icon])=><NavLink key={to} to={to} end={to==='/' } title={label} aria-label={label} className={({isActive})=>`group relative grid h-11 place-items-center rounded-lg transition ${isActive?'bg-teal-400 text-[#10202c]':'text-slate-400 hover:bg-white/10 hover:text-white'}`}><Icon size={19}/><span className="pointer-events-none absolute left-[62px] z-50 hidden whitespace-nowrap rounded-md bg-slate-900 px-2.5 py-1.5 text-xs text-white shadow-lg group-hover:block">{label}</span></NavLink>)}</nav>
      <div className="mb-3 grid place-items-center"><span title={ready?health.data?.storage:'Data unavailable'} className={`h-2.5 w-2.5 rounded-full ring-4 ${ready?'bg-emerald-400 ring-emerald-400/10':'bg-amber-400 ring-amber-400/10'}`}/></div>
    </aside>
    <main className="ml-[72px] min-h-screen">
      <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-slate-200 bg-white px-6">
        <div className="flex items-center gap-3"><div className="text-base font-semibold tracking-tight text-slate-900">DataPilot</div><span className="h-4 w-px bg-slate-200"/><div className="text-xs text-slate-500">Autonomous analytics workspace</div></div>
        <div className={`flex items-center gap-2 rounded-md border px-3 py-1.5 text-xs font-medium ${ready?'border-emerald-200 bg-emerald-50 text-emerald-700':'border-amber-200 bg-amber-50 text-amber-700'}`}><span className={`h-1.5 w-1.5 rounded-full ${ready?'bg-emerald-500':'bg-amber-500'}`}/>{ready?(recovery?'Local recovery data':'Data connected'):'Data unavailable'}</div>
      </header>
      <div className="p-6">{children}</div>
    </main>
  </div>
}
