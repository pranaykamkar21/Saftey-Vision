create table if not exists public.violations (
  id uuid primary key default gen_random_uuid(),
  timestamp_utc timestamptz not null default now(),
  task text not null check (task in ('mask', 'helmet')),
  class_name text not null check (class_name in ('without_mask', 'mask_worn_incorrectly', 'no_helmet')),
  confidence double precision not null check (confidence >= 0.95 and confidence <= 1),
  evidence_path text not null,
  source text not null,
  evidence_kind text not null check (evidence_kind in ('face_crop', 'head_crop')),
  check (
    (task = 'mask' and class_name in ('without_mask', 'mask_worn_incorrectly') and evidence_kind = 'face_crop')
    or (task = 'helmet' and class_name = 'no_helmet' and evidence_kind = 'head_crop')
  )
);

create index if not exists violations_timestamp_desc_idx
  on public.violations (timestamp_utc desc);

create index if not exists violations_task_timestamp_desc_idx
  on public.violations (task, timestamp_utc desc);

alter table public.violations enable row level security;

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('safety-evidence', 'safety-evidence', false, 1048576, array['image/jpeg'])
on conflict (id) do update
set public = false, file_size_limit = 1048576, allowed_mime_types = array['image/jpeg'];
